"""
Modal Service Interface

Handles communication between Railway API and Modal GPU processing.

Author: Modomo Team
Date: November 2025
"""

import os
import modal
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ModalService:
    """
    Interface to Modal serverless GPU processing

    Handles:
    - Submitting jobs to Modal
    - Checking Modal function status
    - Error handling and retries
    """

    def __init__(
        self,
        app_name: Optional[str] = None,
        token_id: Optional[str] = None,
        token_secret: Optional[str] = None
    ):
        """
        Initialize Modal service

        Args:
            app_name: Modal app name (default: from env)
            token_id: Modal token ID (default: from env)
            token_secret: Modal token secret (default: from env)
        """
        self.app_name = app_name or os.getenv("MODAL_APP_NAME") or os.getenv("MODAL_STUB_NAME", "modomo-sd-inference")
        self.token_id = token_id or os.getenv("MODAL_TOKEN_ID")
        self.token_secret = token_secret or os.getenv("MODAL_TOKEN_SECRET")

        if not self.token_id or not self.token_secret:
            raise ValueError(
                "Modal credentials not found. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"
            )

        # Initialize Modal client
        self._init_modal()

        logger.info(f"ModalService initialized with app: {self.app_name}")

    def _init_modal(self):
        """Initialize Modal client and lookup app"""
        try:
            # Set Modal credentials
            os.environ["MODAL_TOKEN_ID"] = self.token_id
            os.environ["MODAL_TOKEN_SECRET"] = self.token_secret

            # Lookup deployed app
            try:
                self.app = modal.App.lookup(self.app_name, create_if_missing=False)
                logger.info(f"✓ Connected to Modal app: {self.app_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not lookup Modal app: {e}")
                logger.warning("Modal app may not be deployed yet. Deploy with: modal deploy")
                self.app = None

        except Exception as e:
            logger.error(f"✗ Failed to initialize Modal: {e}")
            raise

    async def submit_transformation(
        self,
        job_id: str,
        image_url: str,
        style: str,
        room_type: Optional[str] = None,
        preferences: Optional[Dict] = None,
        redis_url: str = None
    ) -> str:
        """
        Submit transformation job to Modal

        Args:
            job_id: Job UUID from Redis
            image_url: R2 URL of uploaded image
            style: Selected style name
            room_type: Room type (optional, will be detected)
            preferences: User preferences (optional)
            redis_url: Redis URL for status updates

        Returns:
            modal_call_id: Modal function call ID for tracking

        Raises:
            Exception: If Modal submission fails
        """
        if not self.app:
            raise Exception("Modal app not initialized. Please deploy Modal function first.")

        logger.info(f"Submitting job {job_id} to Modal...")

        try:
            # Create a handle to the deployed function
            # Use the new Modal syntax for calling deployed functions
            process_transformation = modal.Function.lookup(
                self.app_name,
                "CompleteTransformationPipeline.process_transformation_complete"
            )

            # Submit to Modal (spawn async call)
            call = process_transformation.spawn(
                job_id=job_id,
                image_url=image_url,
                style=style,
                room_type=room_type,
                preferences=preferences or {},
                redis_url=redis_url or os.getenv("REDIS_URL")
            )

            # Get call ID for tracking
            modal_call_id = call.object_id

            logger.info(f"✓ Job {job_id} submitted to Modal: {modal_call_id}")

            return modal_call_id

        except Exception as e:
            logger.error(f"✗ Failed to submit job {job_id} to Modal: {e}")
            raise Exception(f"Modal submission failed: {e}")

    async def check_status(self, modal_call_id: str) -> Dict:
        """
        Check status of Modal function call

        Args:
            modal_call_id: Modal function call ID

        Returns:
            Status dictionary with keys: status, progress, error
        """
        if not self.app:
            return {"status": "error", "error": "Modal app not initialized"}

        try:
            # Get function call
            call = modal.FunctionCall.from_id(modal_call_id)

            # Check if completed
            if call.is_finished():
                try:
                    result = call.get()
                    return {
                        "status": "completed",
                        "progress": 1.0,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "status": "failed",
                        "error": str(e)
                    }
            else:
                return {
                    "status": "running",
                    "progress": 0.5  # Modal doesn't provide granular progress
                }

        except Exception as e:
            logger.error(f"Failed to check Modal status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def cancel_job(self, modal_call_id: str) -> bool:
        """
        Cancel running Modal job

        Args:
            modal_call_id: Modal function call ID

        Returns:
            True if cancelled successfully
        """
        if not self.app:
            return False

        try:
            call = modal.FunctionCall.from_id(modal_call_id)
            call.cancel()
            logger.info(f"✓ Cancelled Modal job: {modal_call_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel Modal job: {e}")
            return False

    def is_deployed(self) -> bool:
        """
        Check if Modal app is deployed

        Returns:
            True if app is deployed and accessible
        """
        return self.app is not None

    async def health_check(self) -> Dict:
        """
        Check Modal service health

        Returns:
            Health status dictionary
        """
        return {
            "modal_connected": self.app is not None,
            "app_name": self.app_name,
            "deployed": self.is_deployed()
        }


# Singleton instance
_modal_service = None


def get_modal_service() -> ModalService:
    """
    Get singleton Modal service instance

    Returns:
        ModalService instance
    """
    global _modal_service
    if _modal_service is None:
        _modal_service = ModalService()
    return _modal_service


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_modal_service():
        """Test Modal service"""

        # Initialize service
        service = ModalService()

        # Check health
        health = await service.health_check()
        print(f"Modal health: {health}")

        # Test submission (requires deployed Modal function)
        if service.is_deployed():
            try:
                modal_call_id = await service.submit_transformation(
                    job_id="test-job-123",
                    image_url="https://example.com/test.jpg",
                    style="modern"
                )
                print(f"✓ Submitted to Modal: {modal_call_id}")

                # Check status
                status = await service.check_status(modal_call_id)
                print(f"Status: {status}")

            except Exception as e:
                print(f"✗ Test failed: {e}")
        else:
            print("⚠️ Modal app not deployed. Deploy with: modal deploy")

    # Run test
    asyncio.run(test_modal_service())
