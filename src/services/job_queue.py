"""
Redis Job Queue Service

Manages async transformation jobs with Redis:
- Job creation and tracking
- Status updates
- PubSub for real-time WebSocket notifications
- Job expiration and cleanup

Author: Modomo Team
Date: November 2025
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import redis
from redis.client import PubSub
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class JobStatus:
    """Job status constants"""
    QUEUED = "queued"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    TRANSFORMING = "transforming"
    COMPLETED = "completed"
    FAILED = "failed"


class JobQueue:
    """
    Redis-based job queue for async transformation processing

    Features:
    - Job creation with UUID
    - Status tracking
    - Progress updates
    - PubSub for real-time notifications
    - Automatic expiration (TTL)
    """

    def __init__(
        self,
        redis_url: str = None,
        redis_password: str = None,
        redis_db: int = 0
    ):
        """
        Initialize Redis connection

        Args:
            redis_url: Redis connection URL (default: from env)
            redis_password: Redis password (default: from env)
            redis_db: Redis database number (default: 0)
        """
        # Get from environment if not provided
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD", "")
        self.redis_db = redis_db or int(os.getenv("REDIS_DB", 0))

        # Initialize Redis client
        self._init_redis()

        logger.info(f"JobQueue initialized with Redis at {self.redis_url}")

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            # Parse connection parameters
            connection_kwargs = {
                'db': self.redis_db,
                'decode_responses': True,
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'retry_on_timeout': True
            }

            if self.redis_password:
                connection_kwargs['password'] = self.redis_password

            # Create Redis client
            self.redis = redis.from_url(
                self.redis_url,
                **connection_kwargs
            )

            # Test connection
            self.redis.ping()
            logger.info("✓ Redis connection successful")

        except redis.ConnectionError as e:
            logger.error(f"✗ Failed to connect to Redis: {e}")
            raise Exception(f"Redis connection failed: {e}")

    async def create_job(
        self,
        user_id: str,
        image_path: str,
        style: str,
        room_type: Optional[str] = None,
        preferences: Optional[Dict] = None
    ) -> str:
        """
        Create new transformation job

        Args:
            user_id: User ID (for tracking and rate limiting)
            image_path: Path to uploaded image (S3/R2)
            style: Selected style name
            room_type: Detected or specified room type
            preferences: Optional user preferences

        Returns:
            job_id: UUID for tracking
        """
        job_id = str(uuid.uuid4())

        job_data = {
            'job_id': job_id,
            'user_id': user_id,
            'status': JobStatus.QUEUED,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'image_path': image_path,
            'style': style,
            'room_type': room_type,
            'preferences': preferences or {},
            'progress': 0.0,
            'estimated_time': 15.0,  # seconds
            'result_url': None,
            'error': None,
            'attempts': 0,
            'max_attempts': 3
        }

        # Store in Redis with 1-hour TTL (3600 seconds)
        self.redis.setex(
            f"job:{job_id}",
            3600,
            json.dumps(job_data)
        )

        # Add to processing queue (list)
        self.redis.lpush('job_queue', job_id)

        logger.info(f"Created job {job_id} for user {user_id} (style: {style})")

        return job_id

    async def get_next_job(self) -> Optional[Dict]:
        """
        Get next job from queue (FIFO)

        Returns:
            Job data dictionary or None if queue is empty
        """
        # Pop from right (FIFO: first in, first out)
        job_id = self.redis.rpop('job_queue')

        if not job_id:
            return None

        # Get job data
        job_data = await self.get_job_status(job_id)

        if not job_data or job_data.get('status') == 'not_found':
            logger.warning(f"Job {job_id} not found in Redis, skipping")
            return await self.get_next_job()  # Recursively get next

        return job_data

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Update job status and notify WebSocket listeners

        Args:
            job_id: Job UUID
            status: New status (from JobStatus constants)
            progress: Progress percentage (0.0 - 1.0)
            result_url: CDN URL of transformed image (when completed)
            error: Error message (if failed)
            metadata: Additional metadata to update
        """
        # Get current job data
        job_data_str = self.redis.get(f"job:{job_id}")

        if not job_data_str:
            logger.warning(f"Job {job_id} not found, cannot update status")
            return

        job_data = json.loads(job_data_str)

        # Update fields
        job_data['status'] = status
        job_data['updated_at'] = datetime.utcnow().isoformat()

        if progress is not None:
            job_data['progress'] = progress

        if result_url:
            job_data['result_url'] = result_url

        if error:
            job_data['error'] = error

        if metadata:
            job_data.update(metadata)

        # Calculate estimated time remaining
        if progress and progress > 0:
            elapsed = (
                datetime.utcnow() -
                datetime.fromisoformat(job_data['created_at'])
            ).total_seconds()
            estimated_total = elapsed / progress
            job_data['estimated_time_remaining'] = max(0, estimated_total - elapsed)

        # Save updated data
        self.redis.setex(f"job:{job_id}", 3600, json.dumps(job_data))

        # Publish update for WebSocket subscribers
        update_message = {
            'job_id': job_id,
            'status': status,
            'progress': progress,
            'result_url': result_url,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }

        self.redis.publish(
            f"job_updates:{job_id}",
            json.dumps(update_message)
        )

        logger.info(
            f"Job {job_id} updated: {status} "
            f"({progress * 100 if progress else 0:.0f}%)"
        )

    async def get_job_status(self, job_id: str) -> Dict:
        """
        Retrieve current job status

        Args:
            job_id: Job UUID

        Returns:
            Job data dictionary
        """
        job_data_str = self.redis.get(f"job:{job_id}")

        if not job_data_str:
            return {'status': 'not_found', 'job_id': job_id}

        return json.loads(job_data_str)

    async def mark_job_failed(
        self,
        job_id: str,
        error_message: str,
        retry: bool = False
    ):
        """
        Mark job as failed and optionally retry

        Args:
            job_id: Job UUID
            error_message: Error description
            retry: Whether to retry the job
        """
        job_data = await self.get_job_status(job_id)

        if job_data.get('status') == 'not_found':
            logger.warning(f"Cannot mark job {job_id} as failed: not found")
            return

        attempts = job_data.get('attempts', 0) + 1
        max_attempts = job_data.get('max_attempts', 3)

        if retry and attempts < max_attempts:
            # Retry: put back in queue
            logger.info(f"Retrying job {job_id} (attempt {attempts}/{max_attempts})")
            await self.update_job_status(
                job_id,
                JobStatus.QUEUED,
                progress=0.0,
                error=f"Retry {attempts}/{max_attempts}: {error_message}",
                metadata={'attempts': attempts}
            )
            self.redis.lpush('job_queue', job_id)
        else:
            # Final failure
            logger.error(f"Job {job_id} failed after {attempts} attempts: {error_message}")
            await self.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=error_message,
                metadata={'attempts': attempts}
            )

    def subscribe_to_job_updates(self, job_id: str) -> PubSub:
        """
        Subscribe to job updates for WebSocket streaming

        Args:
            job_id: Job UUID

        Returns:
            Redis PubSub object
        """
        pubsub = self.redis.pubsub()
        pubsub.subscribe(f"job_updates:{job_id}")
        return pubsub

    async def get_queue_length(self) -> int:
        """
        Get number of jobs waiting in queue

        Returns:
            Queue length
        """
        return self.redis.llen('job_queue')

    async def get_user_jobs(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get recent jobs for a user

        Args:
            user_id: User ID
            limit: Maximum number of jobs to return

        Returns:
            List of job data dictionaries
        """
        # Scan for user's jobs (this is inefficient at scale,
        # consider adding user_jobs:{user_id} list in production)
        jobs = []

        # Get all job keys
        for key in self.redis.scan_iter("job:*"):
            job_data_str = self.redis.get(key)
            if job_data_str:
                job_data = json.loads(job_data_str)
                if job_data.get('user_id') == user_id:
                    jobs.append(job_data)

                    if len(jobs) >= limit:
                        break

        # Sort by created_at (most recent first)
        jobs.sort(
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )

        return jobs[:limit]

    async def cleanup_expired_jobs(self, max_age_hours: int = 24):
        """
        Clean up old jobs (beyond TTL)

        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned = 0

        for key in self.redis.scan_iter("job:*"):
            job_data_str = self.redis.get(key)
            if job_data_str:
                job_data = json.loads(job_data_str)
                created_at = datetime.fromisoformat(job_data['created_at'])

                if created_at < cutoff:
                    self.redis.delete(key)
                    cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired jobs")

    async def get_stats(self) -> Dict:
        """
        Get queue statistics

        Returns:
            Dictionary with queue stats
        """
        queue_length = await self.get_queue_length()

        # Count jobs by status
        status_counts = {
            JobStatus.QUEUED: 0,
            JobStatus.ANALYZING: 0,
            JobStatus.GENERATING: 0,
            JobStatus.TRANSFORMING: 0,
            JobStatus.COMPLETED: 0,
            JobStatus.FAILED: 0
        }

        total_jobs = 0

        for key in self.redis.scan_iter("job:*"):
            job_data_str = self.redis.get(key)
            if job_data_str:
                job_data = json.loads(job_data_str)
                status = job_data.get('status')
                if status in status_counts:
                    status_counts[status] += 1
                total_jobs += 1

        return {
            'queue_length': queue_length,
            'total_jobs': total_jobs,
            'status_counts': status_counts,
            'redis_memory_used': self.redis.info('memory').get('used_memory_human', 'N/A')
        }


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_job_queue():
        """Test job queue functionality"""

        # Initialize queue
        queue = JobQueue()

        # Create test job
        job_id = await queue.create_job(
            user_id="test_user_123",
            image_path="s3://bucket/uploads/test.jpg",
            style="modern",
            room_type="living_room"
        )

        print(f"Created job: {job_id}")

        # Get job status
        status = await queue.get_job_status(job_id)
        print(f"Initial status: {status}")

        # Simulate processing
        await queue.update_job_status(job_id, JobStatus.ANALYZING, progress=0.2)
        await asyncio.sleep(1)

        await queue.update_job_status(job_id, JobStatus.GENERATING, progress=0.5)
        await asyncio.sleep(1)

        await queue.update_job_status(job_id, JobStatus.TRANSFORMING, progress=0.8)
        await asyncio.sleep(1)

        await queue.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=1.0,
            result_url="https://cdn.example.com/result.jpg"
        )

        # Get final status
        final_status = await queue.get_job_status(job_id)
        print(f"Final status: {final_status}")

        # Get stats
        stats = await queue.get_stats()
        print(f"Queue stats: {stats}")

    # Run test
    asyncio.run(test_job_queue())
