"""
WebSocket Manager

Manages WebSocket connections for real-time progress updates.

Author: Modomo Team
Date: November 2025
"""

import json
import asyncio
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and Redis PubSub for real-time updates

    Features:
    - Multiple connections per job (multiple clients)
    - Automatic reconnection handling
    - Redis PubSub subscription
    - Message broadcasting
    """

    def __init__(self, redis_url: str = None):
        """
        Initialize WebSocket manager

        Args:
            redis_url: Redis connection URL (default: from env)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.pubsub_tasks: Dict[str, asyncio.Task] = {}

        logger.info(f"WebSocketManager initialized with Redis: {self.redis_url}")

    async def connect(self, job_id: str, websocket: WebSocket):
        """
        Accept WebSocket connection and subscribe to job updates

        Args:
            job_id: Job UUID to subscribe to
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()

        # Add to active connections
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

        logger.info(f"✓ WebSocket connected for job {job_id} ({len(self.active_connections[job_id])} total)")

        # Start Redis PubSub listener if not already running
        if job_id not in self.pubsub_tasks:
            task = asyncio.create_task(self._subscribe_to_redis(job_id))
            self.pubsub_tasks[job_id] = task

        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": "Connected to job updates"
        })

    async def disconnect(self, job_id: str, websocket: WebSocket):
        """
        Remove WebSocket connection

        Args:
            job_id: Job UUID
            websocket: FastAPI WebSocket instance
        """
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)

                # If no more connections, cancel PubSub task
                if len(self.active_connections[job_id]) == 0:
                    del self.active_connections[job_id]

                    if job_id in self.pubsub_tasks:
                        self.pubsub_tasks[job_id].cancel()
                        del self.pubsub_tasks[job_id]

                    logger.info(f"✓ All WebSockets disconnected for job {job_id}")
                else:
                    logger.info(f"✓ WebSocket disconnected for job {job_id} ({len(self.active_connections[job_id])} remaining)")

    async def _subscribe_to_redis(self, job_id: str):
        """
        Subscribe to Redis PubSub for job updates

        Args:
            job_id: Job UUID
        """
        try:
            # Create Redis PubSub connection
            redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            pubsub = redis_client.pubsub()

            # Subscribe to job updates channel
            channel = f"job_updates:{job_id}"
            await pubsub.subscribe(channel)

            logger.info(f"✓ Subscribed to Redis channel: {channel}")

            # Listen for messages
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Parse update
                        update = json.loads(message["data"])

                        # Broadcast to all connected clients
                        await self._broadcast(job_id, update)

                        # If job completed or failed, stop listening
                        if update.get("status") in ["completed", "failed"]:
                            logger.info(f"Job {job_id} finished, closing Redis subscription")
                            break

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Redis message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")

            # Cleanup
            await pubsub.unsubscribe(channel)
            await redis_client.close()

        except asyncio.CancelledError:
            logger.info(f"Redis subscription cancelled for job {job_id}")
        except Exception as e:
            logger.error(f"Redis subscription error for job {job_id}: {e}")

    async def _broadcast(self, job_id: str, message: dict):
        """
        Broadcast message to all connected clients for a job

        Args:
            job_id: Job UUID
            message: Message dictionary to send
        """
        if job_id not in self.active_connections:
            return

        # Send to all connections
        disconnected = []
        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            await self.disconnect(job_id, websocket)

    async def send_update(self, job_id: str, update: dict):
        """
        Manually send update to connected clients

        Args:
            job_id: Job UUID
            update: Update dictionary
        """
        await self._broadcast(job_id, update)

    def get_connection_count(self, job_id: str) -> int:
        """
        Get number of active connections for a job

        Args:
            job_id: Job UUID

        Returns:
            Number of active WebSocket connections
        """
        return len(self.active_connections.get(job_id, []))

    def get_total_connections(self) -> int:
        """
        Get total number of active connections across all jobs

        Returns:
            Total number of active WebSocket connections
        """
        return sum(len(conns) for conns in self.active_connections.values())

    async def health_check(self) -> Dict:
        """
        Check WebSocket manager health

        Returns:
            Health status dictionary
        """
        return {
            "active_jobs": len(self.active_connections),
            "total_connections": self.get_total_connections(),
            "pubsub_tasks": len(self.pubsub_tasks)
        }


# Singleton instance
_websocket_manager = None


def get_websocket_manager() -> WebSocketManager:
    """
    Get singleton WebSocket manager instance

    Returns:
        WebSocketManager instance
    """
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager


# Example usage with FastAPI
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()
ws_manager = get_websocket_manager()

@app.websocket("/ws/transform/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)

    try:
        # Keep connection alive
        while True:
            # Can receive messages from client if needed
            data = await websocket.receive_text()

            # Echo back (optional)
            await websocket.send_text(f"Message received: {data}")

    except WebSocketDisconnect:
        await ws_manager.disconnect(job_id, websocket)
"""


# Test example
if __name__ == "__main__":
    import asyncio

    async def test_websocket_manager():
        """Test WebSocket manager with mock Redis"""

        manager = WebSocketManager()

        # Check health
        health = await manager.health_check()
        print(f"WebSocket manager health: {health}")

        print("✓ WebSocket manager initialized successfully")

    # Run test
    asyncio.run(test_websocket_manager())
