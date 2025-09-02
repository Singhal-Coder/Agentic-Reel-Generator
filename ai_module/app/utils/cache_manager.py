# 📁 ai_module/utils/cache_manager.py
import hashlib
import httpx
from pathlib import Path
from typing import Optional
import mimetypes

from ..config.settings import settings
from ..utils.logging import logger

class AssetCacheManager:
    """
    Manages downloading and caching any file type (images, videos, audio)
    from URLs to a local directory.
    """
    def __init__(self, cache_dir: Path = settings.CACHE_DIR):
        self.cache_dir = cache_dir
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_cache_path(self, url: str, content_type: Optional[str] = None, ext: Optional[str] = None) -> Path:
        """
        Creates a unique, filesystem-safe filename by hashing the URL.
        """
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # Determine a safe file extension from the Content-Type header if possible
        extension = ext or mimetypes.guess_extension(content_type) if content_type else '' or Path(url).suffix or ''
        # Fallback to the URL's extension if the header is missing
        if not extension:
            if content_type == "sticker":
                extension = ".webp"
            elif content_type == "image":
                extension = ".jpg"
            elif content_type == "video":
                extension = ".mp4"
            elif content_type == "audio":
                extension = ".mpeg"

        dir = self.cache_dir / content_type
        dir.mkdir(parents=True, exist_ok=True)
        return dir / f"{url_hash}{extension}"

    async def get_asset(self, asset_url: str, content_type: Optional[str] = None) -> Optional[Path]:
        """
        Retrieves an asset. Checks the cache first, and if not found,
        downloads it from the URL and saves it to the cache.
        Returns the local Path object to the cached asset.
        """
        # We don't know the content type yet, so we create a preliminary path
        # to check existence. We'll rename it after download if needed.
        headers = {
            "X-API-Key": settings.MUSIC_API_KEY,
            "Content-Type": "application/json"
        }
        preliminary_path = self._get_cache_path(asset_url, content_type)

        if preliminary_path.exists():
            logger.debug(f"Cache HIT for {asset_url}. Using local path: {preliminary_path}")
            return preliminary_path
        logger.info(f"Cache MISS for {asset_url}. Downloading...")
        
        try:
            async with self.async_client.stream("GET", asset_url, headers=headers, follow_redirects=True) as response:
                response.raise_for_status()
                
                content_type_with_extension = response.headers.get('Content-Type')
                if content_type_with_extension:
                    ext = f".{content_type_with_extension.split('/')[-1]}"
                else:
                    ext = None

                final_path = self._get_cache_path(asset_url, content_type, ext)
                
                # If a file with the wrong extension was already guessed, check again
                if final_path.exists():
                    logger.debug(f"Cache HIT for {asset_url}. Using local path: {final_path}")
                    return final_path
                # Write the downloaded content to the cache file
                with open(final_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded and cached asset: {final_path}")
                return final_path

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading {asset_url}: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Network error downloading {asset_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while caching {asset_url}: {e}")
            return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.async_client.aclose()