# =============================================================================
# Handles downloading a single image URL and saving it to the correct
# class subfolder after validation and deduplication.
#
# Workflow per image URL:
#   1. HTTP GET the image bytes (with timeout)
#   2. Save to a temporary file
#   3. Run quality validation (image_validator.is_valid_image)
#   4. Run duplicate check   (image_validator.is_duplicate)
#   5. If both pass → move to final class folder with a clean filename
#   6. If either fails → delete temp file, log the rejection reason
# =============================================================================

import os
import uuid
import time
import shutil
import tempfile
import requests

from image_validator import is_valid_image, is_duplicate


# Browser-like User-Agent header to avoid being blocked by image hosts
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _get_extension(url: str, content_type: str) -> str:
    """
    Determine the appropriate file extension from the URL or HTTP Content-Type.

    Priority: URL extension → Content-Type header → default to .jpg

    Parameters
    ----------
    url : str
        The image URL (may or may not end with an extension).
    content_type : str
        The HTTP Content-Type header value (e.g. "image/jpeg").

    Returns
    -------
    str
        File extension including the dot (e.g. ".jpg", ".png", ".webp").
    """
    # Try to extract extension from the URL path
    url_path = url.split("?")[0]   # Strip query parameters
    url_ext = os.path.splitext(url_path)[-1].lower()
    if url_ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        return url_ext if url_ext != ".jpeg" else ".jpg"

    # Fall back to Content-Type header
    type_map = {
        "image/jpeg": ".jpg",
        "image/png":  ".png",
        "image/webp": ".webp",
        "image/gif":  ".gif",
    }
    for mime, ext in type_map.items():
        if mime in content_type:
            return ext

    # Default — most web images are JPEG
    return ".jpg"


def download_image(
    url: str,
    class_name: str,
    class_dir: str,
    seen_hashes: list,
    config,
    query: str = "",
) -> dict:
    """
    Download one image, validate it, and save it to the class folder if accepted.

    Parameters
    ----------
    url : str
        Direct URL to the image file.
    class_name : str
        The gesture class label (used for filename prefix and logging).
    class_dir : str
        Full path to the class subfolder under dataset_raw/.
    seen_hashes : list
        Accumulated perceptual hashes from all previously accepted images.
        Updated in-place when a new unique image is accepted.
    config : module
        The config module for thresholds and delay settings.
    query : str
        The search query that produced this URL — passed through to the
        result dict so reporter.py can log it in the CSV for your audit.

    Returns
    -------
    dict
        Result record with keys:
          - url, class_name, status ("accepted"/"rejected"), reason, saved_path
        This dict is passed to reporter.py for CSV logging.
    """

    # Template result dict — filled in as we progress through the pipeline
    result = {
        "query":      query,
        "url":        url,
        "class_name": class_name,
        "status":     "rejected",
        "reason":     "",
        "saved_path": "",
    }

    # --- Step 1: HTTP download to a temp file ---
    tmp_path = None
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=config.DOWNLOAD_TIMEOUT_SECONDS,
            stream=True,      # Stream to avoid loading large files into memory
        )

        # Only accept HTTP 200 responses
        if response.status_code != 200:
            result["reason"] = f"http_{response.status_code}"
            return result

        # Determine file extension before writing
        content_type = response.headers.get("Content-Type", "")
        extension = _get_extension(url, content_type)

        # Write to a temporary file (auto-cleaned on failure)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=extension)
        with os.fdopen(tmp_fd, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    except requests.exceptions.Timeout:
        result["reason"] = "download_timeout"
        return result
    except requests.exceptions.ConnectionError:
        result["reason"] = "connection_error"
        return result
    except Exception as e:
        result["reason"] = f"download_error_{str(e)[:40]}"
        return result

    # --- Step 2: Quality validation ---
    valid, reason = is_valid_image(tmp_path, config)
    if not valid:
        os.remove(tmp_path)
        result["reason"] = reason
        return result

    # --- Step 3: Duplicate check ---
    if is_duplicate(tmp_path, seen_hashes, config):
        os.remove(tmp_path)
        result["reason"] = "duplicate"
        return result

    # --- Step 4: Move to final destination ---
    # Filename: <class_name>_<short_uuid><extension>
    # UUID avoids collisions when running multiple queries for the same class
    short_id = uuid.uuid4().hex[:8]
    filename = f"{class_name}_{short_id}{extension}"
    final_path = os.path.join(class_dir, filename)

    try:
        shutil.move(tmp_path, final_path)
    except Exception as e:
        os.remove(tmp_path)
        result["reason"] = f"save_error_{str(e)[:40]}"
        return result

    # --- All checks passed ---
    result["status"]     = "accepted"
    result["reason"]     = "ok"
    result["saved_path"] = final_path

    # Small delay between downloads to be a polite scraper
    time.sleep(config.REQUEST_DELAY_SECONDS)

    return result