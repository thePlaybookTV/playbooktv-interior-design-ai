"""
Cloudflare R2 Image Downloader
Downloads images from Cloudflare R2 bucket for training
"""

import os
import boto3
from botocore.client import Config
from pathlib import Path
from typing import List, Optional, Dict
from tqdm.auto import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CloudflareR2Downloader:
    """Download images from Cloudflare R2 bucket"""

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        """
        Initialize R2 downloader

        Args:
            account_id: Cloudflare account ID
            access_key_id: R2 access key ID
            secret_access_key: R2 secret access key
            bucket_name: R2 bucket name
            endpoint_url: Custom endpoint URL (optional)
        """
        # Load from environment if not provided
        self.account_id = account_id or os.getenv('CLOUDFLARE_ACCOUNT_ID')
        self.access_key_id = access_key_id or os.getenv('R2_ACCESS_KEY_ID')
        self.secret_access_key = secret_access_key or os.getenv('R2_SECRET_ACCESS_KEY')
        self.bucket_name = bucket_name or os.getenv('R2_BUCKET_NAME')

        if not all([self.account_id, self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError(
                "Missing required credentials. Provide via parameters or environment variables:\n"
                "  - CLOUDFLARE_ACCOUNT_ID\n"
                "  - R2_ACCESS_KEY_ID\n"
                "  - R2_SECRET_ACCESS_KEY\n"
                "  - R2_BUCKET_NAME"
            )

        # Construct endpoint URL
        if endpoint_url:
            self.endpoint_url = endpoint_url
        else:
            self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        # Initialize S3 client (R2 is S3-compatible)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'  # R2 uses 'auto' for region
        )

        print(f"✅ Connected to R2 bucket: {self.bucket_name}")

    def test_connection(self) -> bool:
        """Test connection to R2 bucket"""
        try:
            # Try to list objects (limit to 1 for speed)
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            print("✅ Connection test successful!")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def list_images(
        self,
        prefix: str = "",
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp'],
        max_keys: Optional[int] = None
    ) -> List[str]:
        """
        List all images in the bucket

        Args:
            prefix: Filter by prefix (folder path)
            extensions: Valid image extensions
            max_keys: Maximum number of keys to return (None for all)

        Returns:
            List of image keys
        """
        print(f"\n📋 Listing images from bucket: {self.bucket_name}")
        if prefix:
            print(f"   Prefix: {prefix}")

        image_keys = []
        continuation_token = None

        try:
            while True:
                # List objects
                kwargs = {
                    'Bucket': self.bucket_name,
                    'Prefix': prefix,
                    'MaxKeys': 1000
                }

                if continuation_token:
                    kwargs['ContinuationToken'] = continuation_token

                response = self.s3_client.list_objects_v2(**kwargs)

                # Filter for images
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if any(key.lower().endswith(ext) for ext in extensions):
                            image_keys.append(key)

                            # Check max_keys limit
                            if max_keys and len(image_keys) >= max_keys:
                                print(f"   Found {len(image_keys)} images (limit reached)")
                                return image_keys

                # Check if there are more objects
                if response.get('IsTruncated'):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break

            print(f"   Found {len(image_keys)} images")
            return image_keys

        except Exception as e:
            print(f"❌ Error listing images: {e}")
            return []

    def download_image(
        self,
        key: str,
        output_dir: Path,
        keep_structure: bool = True
    ) -> Optional[Path]:
        """
        Download a single image

        Args:
            key: Object key in R2
            output_dir: Local output directory
            keep_structure: Keep folder structure from bucket

        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            # Determine output path
            if keep_structure:
                output_path = output_dir / key
            else:
                output_path = output_dir / Path(key).name

            # Create parent directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                key,
                str(output_path)
            )

            return output_path

        except Exception as e:
            print(f"❌ Failed to download {key}: {e}")
            return None

    def download_images(
        self,
        keys: List[str],
        output_dir: str | Path,
        keep_structure: bool = True,
        max_workers: int = 10,
        skip_existing: bool = True
    ) -> Dict[str, Path]:
        """
        Download multiple images in parallel

        Args:
            keys: List of object keys to download
            output_dir: Local output directory
            keep_structure: Keep folder structure from bucket
            max_workers: Number of parallel download threads
            skip_existing: Skip files that already exist locally

        Returns:
            Dictionary mapping keys to local file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📥 Downloading {len(keys)} images...")
        print(f"   Output: {output_dir}")
        print(f"   Workers: {max_workers}")
        print(f"   Skip existing: {skip_existing}")

        downloaded_files = {}
        skipped = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            futures = {}
            for key in keys:
                # Check if file exists
                if keep_structure:
                    local_path = output_dir / key
                else:
                    local_path = output_dir / Path(key).name

                if skip_existing and local_path.exists():
                    downloaded_files[key] = local_path
                    skipped += 1
                    continue

                future = executor.submit(self.download_image, key, output_dir, keep_structure)
                futures[future] = key

            # Process completed downloads
            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        result = future.result()
                        if result:
                            downloaded_files[key] = result
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"\n❌ Error downloading {key}: {e}")
                        failed += 1

                    pbar.update(1)

        print(f"\n✅ Download complete!")
        print(f"   Downloaded: {len(downloaded_files) - skipped}")
        print(f"   Skipped (existing): {skipped}")
        print(f"   Failed: {failed}")

        return downloaded_files

    def download_all_images(
        self,
        output_dir: str | Path,
        prefix: str = "",
        keep_structure: bool = True,
        max_workers: int = 10,
        skip_existing: bool = True,
        max_images: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        Download all images from bucket

        Args:
            output_dir: Local output directory
            prefix: Filter by prefix (folder path)
            keep_structure: Keep folder structure from bucket
            max_workers: Number of parallel download threads
            skip_existing: Skip files that already exist locally
            max_images: Maximum number of images to download (None for all)

        Returns:
            Dictionary mapping keys to local file paths
        """
        # List images
        keys = self.list_images(prefix=prefix, max_keys=max_images)

        if not keys:
            print("⚠️  No images found to download")
            return {}

        # Download images
        return self.download_images(
            keys=keys,
            output_dir=output_dir,
            keep_structure=keep_structure,
            max_workers=max_workers,
            skip_existing=skip_existing
        )

    def get_public_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for an object

        Args:
            key: Object key
            expires_in: URL expiration time in seconds

        Returns:
            Presigned URL or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            print(f"❌ Failed to generate URL for {key}: {e}")
            return None

    def get_object_info(self, key: str) -> Optional[Dict]:
        """
        Get metadata about an object

        Args:
            key: Object key

        Returns:
            Object metadata dictionary
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return {
                'key': key,
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'etag': response.get('ETag', '').strip('"')
            }
        except Exception as e:
            print(f"❌ Failed to get info for {key}: {e}")
            return None


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download images from Cloudflare R2')
    parser.add_argument('--output', '-o', type=str, default='./r2_images',
                        help='Output directory for downloaded images')
    parser.add_argument('--prefix', '-p', type=str, default='',
                        help='Bucket prefix/folder to download from')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to download')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='Number of parallel download workers')
    parser.add_argument('--test', action='store_true',
                        help='Test connection only')
    parser.add_argument('--list', action='store_true',
                        help='List images only (no download)')

    args = parser.parse_args()

    print("=" * 70)
    print("🌐 CLOUDFLARE R2 IMAGE DOWNLOADER")
    print("=" * 70)

    try:
        # Initialize downloader
        downloader = CloudflareR2Downloader()

        # Test connection
        if args.test:
            downloader.test_connection()
            exit(0)

        # List images
        if args.list:
            images = downloader.list_images(prefix=args.prefix, max_keys=args.max_images)
            print(f"\n📋 Found {len(images)} images:")
            for i, key in enumerate(images[:20], 1):
                print(f"   {i}. {key}")
            if len(images) > 20:
                print(f"   ... and {len(images) - 20} more")
            exit(0)

        # Download images
        downloaded = downloader.download_all_images(
            output_dir=args.output,
            prefix=args.prefix,
            max_images=args.max_images,
            max_workers=args.workers,
            skip_existing=True
        )

        print("\n" + "=" * 70)
        print(f"✅ Downloaded {len(downloaded)} images to {args.output}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
