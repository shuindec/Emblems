# =============================================================================
# reporter.py
# -----------------------------------------------------------------------------
# Handles all logging and reporting for the scraping run.
#
# Two responsibilities:
#   1. CSV report  — writes one row per image URL attempted, with outcome
#   2. Summary     — prints a human-readable count table at the end of the run
#
# The CSV report lets you audit which images were rejected and why,
# making it easy to adjust thresholds in config.py if needed.
# =============================================================================

import os
import csv
from datetime import datetime


# CSV column headers — one row written per attempted image download
# "query" is included so you can audit which search query produced each image
# and manually fill "is_classified_correctly" and "move_to_class" afterwards
CSV_COLUMNS = [
    "timestamp",    # When this image was processed
    "query",        # The search query that produced this image URL
    "class_name",   # Gesture class label the image was saved under
    "url",          # Source image URL
    "status",       # "accepted" or "rejected"
    "reason",       # "ok" if accepted, rejection reason string if rejected
    "saved_path",   # Final file path (empty if rejected)
]


def init_report(report_path: str) -> None:
    """
    Initialise the CSV report file for a scraping run.

    Behaviour:
      - If the file does NOT exist → create it and write the header row
      - If the file ALREADY exists → append to it without rewriting the header

    This preserves all previous round data and your manual audit columns
    (is_classified_correctly, move_to_class) across multiple scraping runs.
    New rows from round 2 simply appear below round 1 rows in the same file.

    Parameters
    ----------
    report_path : str
        Full path to the CSV file (from config.REPORT_PATH).
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Only write the header if this is a brand new file
    file_exists = os.path.isfile(report_path)

    with open(report_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            # New file — write header so Excel/pandas can read it correctly
            writer.writeheader()


def log_result(report_path: str, result: dict) -> None:
    """
    Append one image result row to the CSV report.

    Called after every download attempt (accepted or rejected).
    Appends rather than rewrites, so the file grows incrementally
    and is not lost if the script crashes mid-run.

    Parameters
    ----------
    report_path : str
        Full path to the CSV file.
    result : dict
        The result dict returned by downloader.download_image(),
        with keys: query, url, class_name, status, reason, saved_path.
    """
    row = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query":      result.get("query", ""),        # which query produced this image
        "class_name": result.get("class_name", ""),
        "url":        result.get("url", ""),
        "status":     result.get("status", ""),
        "reason":     result.get("reason", ""),
        "saved_path": result.get("saved_path", ""),
    }

    # Open in append mode so each result is written immediately
    with open(report_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def print_summary(class_counts: dict, total_target: int) -> None:
    """
    Print a formatted summary table at the end of the scraping run.

    Shows accepted and rejected counts per class, and the overall
    progress toward the total image target.

    Parameters
    ----------
    class_counts : dict
        Keys are class names. Values are dicts with "accepted" and "rejected"
        integer counts. Example:
        {
            "2-finger-heart": {"accepted": 47, "rejected": 12},
            "traditional-heart": {"accepted": 55, "rejected": 8},
            ...
        }
    total_target : int
        The TOTAL_IMAGE_TARGET from config — used to show progress.
    """
    print("\n" + "=" * 60)
    print("SCRAPE RUN SUMMARY")
    print("=" * 60)

    total_accepted = 0
    total_rejected = 0

    # Print per-class breakdown
    print(f"{'Class':<25} {'Accepted':>10} {'Rejected':>10}")
    print("-" * 50)

    for class_name, counts in class_counts.items():
        accepted = counts.get("accepted", 0)
        rejected = counts.get("rejected", 0)
        total_accepted += accepted
        total_rejected += rejected
        print(f"{class_name:<25} {accepted:>10} {rejected:>10}")

    print("-" * 50)
    print(f"{'TOTAL':<25} {total_accepted:>10} {total_rejected:>10}")
    print()
    print(f"Target : {total_target} images")
    print(f"Saved  : {total_accepted} images")
    print(f"Progress: {min(100, round(total_accepted / total_target * 100))}%")
    print("=" * 60)