# =============================================================================
# web_image_scraper.py
# -----------------------------------------------------------------------------
# Main entry point for the web image scraping pipeline.
#
# Run this file to start scraping:
#   python web_image_scraper.py
#
# Before running:
#   1. Open config.py and paste your GOOGLE_API_KEY
#   2. Confirm BASE_DIR path matches your machine
#   3. Adjust CLASS_QUERIES or thresholds in config.py if needed
#
# What this script does:
#   - Loops over each gesture class and its search queries (from config.py)
#   - For each query, fetches image URLs via Google Custom Search API
#   - Downloads, validates, and deduplicates each image
#   - Saves accepted images to outputs/dataset_raw/<class>/
#   - Logs every attempt (accepted or rejected) to outputs/logs/scrape_report.csv
#   - Stops early once TOTAL_IMAGE_TARGET images have been accepted
# =============================================================================

import os
import sys
from tqdm import tqdm

# Local modules — all in the same folder
import config
from google_search import build_search_service, search_images
from downloader import download_image
from reporter import init_report, log_result, print_summary


# =============================================================================
# FOLDER SETUP
# Creates all required output directories before the run starts.
# Safe to run multiple times — os.makedirs with exist_ok won't overwrite.
# =============================================================================
def setup_directories() -> dict:
    """
    Create the output folder structure under config.DATASET_DIR and config.LOG_DIR.
    Returns
    -------
    dict
        Maps each class name to its full folder path, e.g.:
        {"2-finger-heart": "C:\\...\\dataset_raw\\2-finger-heart", ...}
    """
    class_dirs = {}
 
    for class_name in config.CLASS_QUERIES:
        # Replace hyphens with underscores in folder names for OS compatibility
        folder_name = class_name.replace(" ", "_")
        class_path = os.path.join(config.DATASET_DIR, folder_name)
        os.makedirs(class_path, exist_ok=True)
        class_dirs[class_name] = class_path
 
    # Create logs directory
    os.makedirs(config.LOG_DIR, exist_ok=True)
 
    return class_dirs
 
 
# =============================================================================
# VALIDATION CHECK
# Catches missing API credentials before any API calls are made.
# =============================================================================
 
def validate_config() -> None:
    """
    Check that required config values have been filled in.
    Exits with a clear message if the API key placeholder was not replaced.
    """
    if config.SERPER_API_KEY == "YOUR_SERPER_API_KEY_HERE":
        print("[ERROR] Please open config.py and replace SERPER_API_KEY with your actual key.")
        sys.exit(1)
 
 
# =============================================================================
# MAIN PIPELINE
# =============================================================================
 
def main():
    """
    Orchestrate the full scraping pipeline across all classes and queries.
 
    Pipeline per class:
      For each query in CLASS_QUERIES[class_name]:
        1. Fetch image URLs from Google Custom Search API
        2. For each URL → download → validate → dedup → save or skip
        3. Log each attempt to CSV
        4. Stop early if TOTAL_IMAGE_TARGET is reached
    """
 
    print("=" * 60)
    print("WEB IMAGE SCRAPER — Heart Gesture Dataset")
    print("=" * 60)
 
    # --- Pre-flight checks ---
    validate_config()
 
    # --- Set up folder structure ---
    class_dirs = setup_directories()
    print(f"\nOutput folder : {config.DATASET_DIR}")
    print(f"Report file   : {config.REPORT_PATH}")
    print(f"Image target  : {config.TOTAL_IMAGE_TARGET} total across all classes\n")
 
    # --- Initialise CSV report ---
    init_report(config.REPORT_PATH)
 
    # --- Build Serper service (stateless — returns key for interface compatibility) ---
    print("Connecting to Serper Image Search API...")
    try:
        search_service = build_search_service(config.SERPER_API_KEY)
        print("Connected.\n")
    except Exception as e:
        print(f"[ERROR] Could not initialise Serper connection: {e}")
        sys.exit(1)
 
    # --- Tracking state across the full run ---
    total_accepted = 0   # Total images saved across all classes
 
    # Per-class accepted/rejected counts for the summary report
    class_counts = {
        class_name: {"accepted": 0, "rejected": 0}
        for class_name in config.CLASS_QUERIES
    }
 
    # Perceptual hashes of all accepted images — shared across classes
    # so a duplicate image won't be saved under two different class folders
    seen_hashes = []
 
    # ==========================================================================
    # OUTER LOOP: iterate over each gesture class
    # ==========================================================================
    for class_name, queries in config.CLASS_QUERIES.items():
 
        # Stop if overall target already reached
        if total_accepted >= config.TOTAL_IMAGE_TARGET:
            print(f"\nTotal target of {config.TOTAL_IMAGE_TARGET} images reached. Stopping.")
            break
 
        class_dir = class_dirs[class_name]
        print(f"{'─' * 50}")
        print(f"Class: {class_name}")
        print(f"Folder: {class_dir}")
        print(f"Queries to run: {len(queries)}")
 
        # ======================================================================
        # INNER LOOP: iterate over each search query for this class
        # ======================================================================
        for query_idx, query in enumerate(queries, start=1):
 
            if total_accepted >= config.TOTAL_IMAGE_TARGET:
                break
 
            print(f"\n  Query {query_idx}/{len(queries)}: \"{query}\"")
 
            # Step 1 — Fetch image URLs from Serper Google Images API
            image_results = search_images(
                query=query,
                config=config,
            )
 
            if not image_results:
                print(f"  [WARNING] No results returned for this query.")
                continue
 
            print(f"  Found {len(image_results)} URLs. Downloading...")
 
            # Step 2 — Download, validate, and save each image URL
            # tqdm wraps the list to display a progress bar per query
            for img_info in tqdm(image_results, desc=f"  {class_name}", unit="img", leave=False):
 
                if total_accepted >= config.TOTAL_IMAGE_TARGET:
                    break
 
                # Attempt to download and validate this image URL
                result = download_image(
                    url=img_info["url"],
                    class_name=class_name,
                    class_dir=class_dir,
                    seen_hashes=seen_hashes,
                    config=config,
                    query=img_info.get("query", query),
                )
 
                # Log every attempt to CSV (accepted and rejected)
                log_result(config.REPORT_PATH, result)
 
                # Update running counts
                if result["status"] == "accepted":
                    total_accepted += 1
                    class_counts[class_name]["accepted"] += 1
                else:
                    class_counts[class_name]["rejected"] += 1
 
            # Show per-query progress
            accepted_so_far = class_counts[class_name]["accepted"]
            print(f"  Accepted so far for '{class_name}': {accepted_so_far}")
 
        print(f"\nClass '{class_name}' complete — {class_counts[class_name]['accepted']} images saved.")
 
    # ==========================================================================
    # END OF RUN — print summary
    # ==========================================================================
    print_summary(class_counts, config.TOTAL_IMAGE_TARGET)
    print(f"\nFull report saved to: {config.REPORT_PATH}")
 
 
# Run the pipeline when this file is executed directly
if __name__ == "__main__":
    main()