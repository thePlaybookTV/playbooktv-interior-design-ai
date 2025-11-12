"""
S3/Cloudflare R2 Storage Service

Manages image uploads and CDN delivery:
- Upload original images, control images, and results
- Generate thumbnails
- Create signed URLs
- CDN optimization

Author: Modomo Team
Date: November 2025
"""

import io
import os
from typing import Optional, Tuple
from PIL import Image
import boto3
from botocore.exceptions import ClientError
import logging
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)


class StorageService:
    """
    S3/Cloudflare R2 storage with CDN integration

    Features:
    - Upload images to R2
    - Generate thumbnails
    - Signed URLs for private access
    - CDN delivery for public images
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        cdn_domain: Optional[str] = None
    ):
        """
        Initialize S3/R2 storage client

        Args:
            bucket_name: S3/R2 bucket name
            endpoint_url: S3/R2 endpoint (R2 uses custom endpoint)
            access_key_id: AWS/R2 access key
            secret_access_key: AWS/R2 secret key
            cdn_domain: CDN domain for public URLs
        """
        # Get from environment if not provided
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME")
        self.endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT_URL")
        self.access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY")
        self.cdn_domain = cdn_domain or os.getenv("CDN_DOMAIN")

        # Validate configuration
        if not all([self.bucket_name, self.endpoint_url, self.access_key_id, self.secret_access_key]):
            raise ValueError(
                "Missing R2 configuration. Please set R2_BUCKET_NAME, "
                "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY"
            )

        # Initialize S3 client
        self._init_s3_client()

        logger.info(f"StorageService initialized with bucket: {self.bucket_name}")

    def _init_s3_client(self):
        """Initialize S3/R2 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name='auto'  # R2 uses 'auto' region
            )

            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info("✓ S3/R2 connection successful")

        except ClientError as e:
            logger.error(f"✗ Failed to connect to S3/R2: {e}")
            raise Exception(f"S3/R2 connection failed: {e}")

    async def upload_image(
        self,
        image: Image.Image,
        key: str,
        optimize: bool = True,
        quality: int = 85,
        make_public: bool = True
    ) -> str:
        """
        Upload image to S3/R2 with optional optimization

        Args:
            image: PIL Image object
            key: S3 object key (path in bucket)
            optimize: Whether to optimize image
            quality: JPEG quality (1-100)
            make_public: Whether to make image publicly accessible

        Returns:
            Public CDN URL or S3 URL
        """
        logger.info(f"Uploading image to {key}...")

        # Optimize image if requested
        if optimize:
            image = self._optimize_image(image, quality)

        # Convert to bytes
        buffer = io.BytesIO()
        image_format = 'JPEG' if image.mode == 'RGB' else 'PNG'
        image.save(buffer, format=image_format, quality=quality, optimize=True)
        buffer.seek(0)

        # Set content type
        content_type = f'image/{image_format.lower()}'

        # Upload to S3/R2
        try:
            extra_args = {
                'ContentType': content_type,
                'CacheControl': 'public, max-age=31536000',  # 1 year
            }

            if make_public:
                extra_args['ACL'] = 'public-read'

            self.s3_client.upload_fileobj(
                buffer,
                self.bucket_name,
                key,
                ExtraArgs=extra_args
            )

            # Generate URL
            if make_public and self.cdn_domain:
                url = f"https://{self.cdn_domain}/{key}"
            else:
                url = f"{self.endpoint_url}/{self.bucket_name}/{key}"

            logger.info(f"✓ Image uploaded successfully to {key}")
            return url

        except ClientError as e:
            logger.error(f"✗ Failed to upload image: {e}")
            raise Exception(f"Image upload failed: {e}")

    async def upload_from_path(
        self,
        file_path: str,
        key: str,
        make_public: bool = True
    ) -> str:
        """
        Upload file from local path

        Args:
            file_path: Local file path
            key: S3 object key
            make_public: Whether to make publicly accessible

        Returns:
            Public URL
        """
        try:
            # Open image
            image = Image.open(file_path)

            # Upload
            return await self.upload_image(
                image,
                key,
                optimize=True,
                make_public=make_public
            )

        except Exception as e:
            logger.error(f"Failed to upload from path {file_path}: {e}")
            raise

    async def upload_transformation_set(
        self,
        job_id: str,
        original_image: Image.Image,
        control_images: dict,
        result_image: Optional[Image.Image] = None
    ) -> dict:
        """
        Upload complete set of images for a transformation job

        Args:
            job_id: Job UUID
            original_image: Original uploaded image
            control_images: Dict of control images (depth, canny, segmentation)
            result_image: Transformed result image (optional, upload later)

        Returns:
            Dictionary of uploaded URLs
        """
        urls = {}

        # Upload original
        urls['original'] = await self.upload_image(
            original_image,
            f"uploads/{job_id}/original.jpg",
            make_public=False  # Keep original private
        )

        # Upload control images
        if control_images.get('depth'):
            urls['depth'] = await self.upload_image(
                control_images['depth'],
                f"uploads/{job_id}/depth.jpg",
                make_public=False
            )

        if control_images.get('canny'):
            urls['canny'] = await self.upload_image(
                control_images['canny'],
                f"uploads/{job_id}/canny.jpg",
                make_public=False
            )

        if control_images.get('segmentation'):
            urls['segmentation'] = await self.upload_image(
                control_images['segmentation'],
                f"uploads/{job_id}/segmentation.jpg",
                make_public=False
            )

        # Upload result if provided
        if result_image:
            # Upload full resolution result
            urls['result'] = await self.upload_image(
                result_image,
                f"results/{job_id}/transformed.jpg",
                quality=95,  # High quality for result
                make_public=True
            )

            # Generate and upload thumbnail
            thumbnail = await self.generate_thumbnail(result_image)
            urls['thumbnail'] = await self.upload_image(
                thumbnail,
                f"results/{job_id}/thumbnail.jpg",
                quality=70,
                make_public=True
            )

        logger.info(f"✓ Uploaded transformation set for job {job_id}")
        return urls

    async def generate_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (480, 360)
    ) -> Image.Image:
        """
        Generate thumbnail for quick preview

        Args:
            image: Input PIL Image
            size: Thumbnail size (width, height)

        Returns:
            Thumbnail image
        """
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.LANCZOS)
        return thumbnail

    def _optimize_image(
        self,
        image: Image.Image,
        quality: int = 85
    ) -> Image.Image:
        """
        Optimize image for web delivery

        Args:
            image: Input PIL Image
            quality: Target quality (1-100)

        Returns:
            Optimized image
        """
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Alpha channel
            image = rgb_image

        # Resize if too large (max 1920x1080)
        max_size = (1920, 1080)
        if image.width > max_size[0] or image.height > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)

        return image

    async def get_signed_url(
        self,
        key: str,
        expires_in: int = 3600
    ) -> str:
        """
        Generate signed URL for private access

        Args:
            key: S3 object key
            expires_in: Expiration time in seconds (default: 1 hour)

        Returns:
            Signed URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key
                },
                ExpiresIn=expires_in
            )
            return url

        except ClientError as e:
            logger.error(f"Failed to generate signed URL: {e}")
            raise

    async def delete_job_images(self, job_id: str):
        """
        Delete all images associated with a job

        Args:
            job_id: Job UUID
        """
        try:
            # List all objects with job_id prefix
            prefixes = [
                f"uploads/{job_id}/",
                f"results/{job_id}/"
            ]

            for prefix in prefixes:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )

                if 'Contents' in response:
                    for obj in response['Contents']:
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )

            logger.info(f"✓ Deleted images for job {job_id}")

        except ClientError as e:
            logger.error(f"Failed to delete job images: {e}")
            raise

    async def get_storage_stats(self) -> dict:
        """
        Get storage statistics

        Returns:
            Dictionary with storage stats
        """
        try:
            # Count objects and total size
            uploads_count = 0
            results_count = 0
            uploads_size = 0
            results_size = 0

            # List uploads
            paginator = self.s3_client.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix='uploads/'):
                if 'Contents' in page:
                    uploads_count += len(page['Contents'])
                    uploads_size += sum(obj['Size'] for obj in page['Contents'])

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix='results/'):
                if 'Contents' in page:
                    results_count += len(page['Contents'])
                    results_size += sum(obj['Size'] for obj in page['Contents'])

            return {
                'uploads_count': uploads_count,
                'results_count': results_count,
                'uploads_size_mb': uploads_size / (1024 * 1024),
                'results_size_mb': results_size / (1024 * 1024),
                'total_size_mb': (uploads_size + results_size) / (1024 * 1024)
            }

        except ClientError as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_storage_service():
        """Test storage service"""

        # Initialize service
        storage = StorageService()

        # Load test image
        test_image = Image.open("test_room.jpg")

        # Upload image
        url = await storage.upload_image(
            test_image,
            f"test/test_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            optimize=True,
            make_public=True
        )

        print(f"✓ Uploaded image: {url}")

        # Generate thumbnail
        thumbnail = await storage.generate_thumbnail(test_image)
        thumb_url = await storage.upload_image(
            thumbnail,
            f"test/test_thumbnail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            make_public=True
        )

        print(f"✓ Uploaded thumbnail: {thumb_url}")

        # Get stats
        stats = await storage.get_storage_stats()
        print(f"Storage stats: {stats}")

    # Run test
    asyncio.run(test_storage_service())
