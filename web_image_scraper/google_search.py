# =============================================================================
# google_search.py
# -----------------------------------------------------------------------------
# Fetches image URLs using the Serper API (https://serper.dev).
# Serper wraps Google Image Search and returns clean structured JSON,
# making it a drop-in replacement for the retired Google Custom Search API.
#
# Key differences from the previous Google CSE implementation:
#   - Uses POST requests with JSON body (not GET with query params)
#   - Authentication via "X-API-KEY" header (not URL parameter)
#   - Returns imageUrl field (not link) for the direct image URL
#   - No pagination needed — Serper returns up to 10 results per call,
#     which matches our RESULTS_PER_QUERY setting of 10
#   - No separate service object to build — just requests.post() each time
#
# The playground result confirmed the exact data structure this module
# parses: title, imageUrl, imageWidth, imageHeight, source, domain
# =============================================================================

import time
import requests


# =============================================================================
# MAIN SEARCH FUNCTION
# =============================================================================

def search_images(query: str, config) -> list[dict]:
    """
    Search for images matching a query using the Serper Google Images API.

    Sends a POST request to Serper's image search endpoint and parses
    the returned JSON into a flat list of image dicts. Each dict contains
    the direct image URL and metadata needed for logging and downloading.

    From the playground test, the Serper response structure is:
    {
        "images": [
            {
                "imageUrl":    "https://...",   <- direct URL to the image file
                "title":       "...",
                "imageWidth":  650,
                "imageHeight": 484,
                "source":      "The Korea Herald",
                "domain":      "www.koreaherald.com",
                "link":        "https://..."    <- page where image was found
            },
            ...
        ],
        "credits": 1   <- number of Serper credits consumed by this call
    }

    Parameters
    ----------
    query : str
        The search query string (e.g. "finger heart kpop idol photo").
    config : module
        The config module, providing SERPER_API_KEY, SERPER_ENDPOINT,
        SERPER_COUNTRY, RESULTS_PER_QUERY, REQUEST_DELAY_SECONDS,
        and DOWNLOAD_TIMEOUT_SECONDS.

    Returns
    -------
    list of dict
        Each dict contains:
          - "url"    : direct image file URL (used by downloader.py)
          - "title"  : image title from Google (for logging)
          - "source" : source website name (for logging)
        Returns an empty list if the API call fails or returns no images.
    """

    # --- Build request headers ---
    # Serper authenticates via header, not URL parameter
    headers = {
        "X-API-KEY":    config.SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    # --- Build request body ---
    # num: number of results (Serper supports up to 10 per call)
    # gl:  country code for search localisation
    payload = {
        "q":   query,
        "num": config.RESULTS_PER_QUERY,
        "gl":  config.SERPER_COUNTRY,
    }

    # --- Make the API call ---
    try:
        response = requests.post(
            config.SERPER_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=config.DOWNLOAD_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        print(f"    [API TIMEOUT] Query '{query}' timed out.")
        return []
    except requests.exceptions.ConnectionError as e:
        print(f"    [API CONNECTION ERROR] {e}")
        return []
    except Exception as e:
        print(f"    [API UNEXPECTED ERROR] {e}")
        return []

    # --- Handle non-200 responses with clear messages ---
    if response.status_code == 401:
        print(f"    [API ERROR 401] Invalid API key — check SERPER_API_KEY in config.py")
        return []
    if response.status_code == 429:
        print(f"    [API ERROR 429] Rate limit hit — increase REQUEST_DELAY_SECONDS in config.py")
        return []
    if response.status_code != 200:
        print(f"    [API ERROR {response.status_code}] Query '{query}': {response.text[:100]}")
        return []

    # --- Parse response JSON ---
    try:
        data = response.json()
    except Exception as e:
        print(f"    [PARSE ERROR] Could not parse API response: {e}")
        return []

    # Images live under the "images" key, as confirmed in the playground test
    raw_images = data.get("images", [])

    if not raw_images:
        print(f"    [WARNING] No images returned for query '{query}'")
        return []

    # --- Normalise into the flat dict format expected by downloader.py ---
    # downloader.py only needs: url, title, source
    results = []
    for item in raw_images:
        image_url = item.get("imageUrl", "")
        if not image_url:
            continue   # skip items missing a direct image URL

        results.append({
            "url":    image_url,
            "title":  item.get("title", ""),
            "source": item.get("source", item.get("domain", "")),
            "query":  query,    # originating query through to the CSV report
        })

    # Polite pause between successive API calls to avoid rate limiting
    time.sleep(config.REQUEST_DELAY_SECONDS)

    return results


# =============================================================================
# COMPATIBILITY SHIM
# -----------------------------------------------------------------------------
# web_image_scraper.py calls build_search_service() once at startup and
# passes the returned object into search_images() as "service".
# Serper is stateless so no persistent object is needed — this stub simply
# returns the API key, and search_images() ignores the "service" argument
# by reading directly from config instead.
# =============================================================================

def build_search_service(api_key: str):
    """
    Compatibility stub — Serper requires no persistent service object.

    Returns the API key unchanged so the main orchestrator script
    needs no modification despite the backend changing from Google CSE
    to Serper.

    Parameters
    ----------
    api_key : str
        Your Serper API key (config.SERPER_API_KEY).

    Returns
    -------
    str
        The same API key, passed through unchanged.
    """
    return api_key