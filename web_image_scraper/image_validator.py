# Validates downloaded images before saving them to the dataset.
# Two responsibilities:
#   1. Quality filtering  — rejects icons, sketches, corrupt files, thumbnails
#   2. Duplicate detection — perceptual hashing to skip near-identical images
#
# Called by downloader.py after each image is downloaded to a temp location.

import os
from PIL import Image
from PIL.Image import Image as PILImage
import PIL
import imagehash

# =============================================================================
# QUALITY VALIDATION
# Checks file size, image dimensions, and aspect ratio.
# Returns (True, "ok") if the image passes, or (False, reason) if rejected.
# =============================================================================

def is_valid_image(img_path: str, config) -> tuple[bool, str]:
    """
    Validate a single image against quality thresholds defined in config.

    Checks performed (in order of cheapest to most expensive):
      1. File exists and is non-empty
      2. File size within acceptable range (rejects tiny icons and huge banners)
      3. Image is openable by Pillow (rejects corrupt/partial downloads)
      4. Pixel dimensions meet minimum size (rejects thumbnails)
      5. Aspect ratio within bounds (rejects banner strips and thin slices)

    Parameters
    ----------
    img_path : str
        Full path to the downloaded image file.
    config : module
        The config module, used for threshold constants.

    Returns
    -------
    tuple[bool, str]
        (True, "ok") if valid, (False, rejection_reason) if not.
    """

    # --- Check 1: File exists ---
    if not os.path.exists(img_path):
        return False, "file_not_found"

    # --- Check 2: File size ---
    file_size = os.path.getsize(img_path)
    if file_size < config.MIN_FILE_SIZE_BYTES:
        return False, f"file_too_small_{file_size}bytes"
    if file_size > config.MAX_FILE_SIZE_BYTES:
        return False, f"file_too_large_{file_size}bytes"

    # --- Check 3: Pillow can open it (catches corrupt downloads) ---
    try:
        img = Image.open(img_path)
        img.verify()  # verify() checks the file header without decoding all pixels
    except Exception as e:
        return False, f"corrupt_or_unreadable_{str(e)[:40]}"

    # Re-open after verify() because verify() leaves the file in a closed state
    try:
        img = Image.open(img_path)
        width, height = img.size
    except Exception as e:
        return False, f"cannot_read_dimensions_{str(e)[:40]}"

    # --- Check 4: Minimum pixel dimensions ---
    if width < config.MIN_WIDTH or height < config.MIN_HEIGHT:
        return False, f"too_small_{width}x{height}px"

    # --- Check 5: Aspect ratio (width / height) ---
    aspect_ratio = width / height
    if aspect_ratio < config.MIN_ASPECT_RATIO:
        return False, f"aspect_too_tall_{aspect_ratio:.2f}"
    if aspect_ratio > config.MAX_ASPECT_RATIO:
        return False, f"aspect_too_wide_{aspect_ratio:.2f}"

    # --- Check 6: Colour complexity (photo vs illustration) ---
    # Re-use the already-opened image object to avoid reading the file again
    real, colour_reason = is_real_photo(img, config)
    if not real:
        return False, colour_reason

    return True, "ok"


# =============================================================================
# COLOUR COMPLEXITY CHECK
# Separates real photographs from illustrations and clipart by counting
# how many distinct colours remain after quantizing the image to a small
# fixed palette.
#
# Why quantize first?
#   A raw photograph may contain millions of unique RGB values due to
#   JPEG noise and lighting gradients — counting all of them directly is
#   slow and noisy. Quantizing to 256 colours first compresses the colour
#   space into a manageable representation while preserving the essential
#   difference between colour-rich photos and flat illustrations.
#
# After quantizing, we convert back to RGB and count unique pixel values.
# This count reflects colour diversity across the whole image:
#   - Real photo    → many varied quantized colours  (typically 200–256)
#   - Illustration  → very few flat quantized colours (typically 5–30)
# =============================================================================

def is_real_photo(img: PILImage, config) -> tuple[bool, str]:
    """
    Check whether an image has sufficient colour complexity to be a real photo.

    Algorithm:
      1. Convert image to RGB (handles RGBA, palette-mode, greyscale inputs)
      2. Quantize to a fixed palette of COLOUR_QUANTIZE_PALETTE colours
         This reduces millions of raw pixel values to at most 256 buckets
      3. Convert the quantized result back to RGB pixel values
      4. Count the number of unique RGB tuples across all pixels
      5. Reject if the count is below MIN_DISTINCT_COLOURS

    Parameters
    ----------
    img : PIL.Image.Image
        Already-opened Pillow image object (passed from is_valid_image
        to avoid re-opening the file).
    config : module
        The config module for COLOUR_QUANTIZE_PALETTE and MIN_DISTINCT_COLOURS.

    Returns
    -------
    tuple[bool, str]
        (True, "ok") if colour complexity is sufficient,
        (False, reason_string) if the image looks like an illustration.
    """
    try:
        # Step 1 — Normalise to RGB
        # RGBA images (PNG with transparency) and palette-mode images need
        # conversion before quantize() works reliably
        rgb_img = img.convert("RGB")

        # Step 2 — Quantize to a small fixed palette
        # This maps every pixel to its nearest colour in a 256-colour palette,
        # compressing the image's colour space into at most 256 buckets
        quantized = rgb_img.quantize(colors=config.COLOUR_QUANTIZE_PALETTE)

        # Step 3 — Convert quantized result back to RGB pixel values
        # quantize() returns a palette-mode image; converting to RGB maps
        # each palette index back to its actual RGB colour tuple
        quantized_rgb = quantized.convert("RGB")

        # Step 4 — Count unique RGB tuples across all pixels
        # getcolors() returns a list of (pixel_count, colour) tuples for every
        # unique colour present. maxcolors is set high enough to never truncate.
        # len() of this list gives the number of distinct colours in the image.
        colour_list = quantized_rgb.getcolors(maxcolors=config.COLOUR_QUANTIZE_PALETTE ** 2)
        unique_colours = len(colour_list) if colour_list else 0

        # Step 5 — Reject if below the complexity threshold
        if unique_colours < config.MIN_DISTINCT_COLOURS:
            return False, f"illustration_low_colours_{unique_colours}"

    except Exception as e:
        # If anything goes wrong during colour analysis, reject to be safe
        return False, f"colour_check_error_{str(e)[:40]}"

    return True, "ok"


# =============================================================================
# DUPLICATE DETECTION
# Uses perceptual hashing (pHash) to detect visually similar images.
# Unlike MD5/SHA hashing, pHash catches re-sized or slightly cropped duplicates
# from different URLs — very common in web image search results.
# =============================================================================

def compute_hash(img_path: str):
    """
    Compute the perceptual hash (pHash) of an image.

    pHash works by:
      1. Resizing the image to a small fixed size (e.g. 32x32)
      2. Converting to grayscale
      3. Applying a DCT (discrete cosine transform)
      4. Encoding the high-frequency components as a bit string

    Two visually similar images will have very similar (low Hamming distance)
    hashes, even if they differ in resolution, compression, or minor cropping.

    Parameters
    ----------
    img_path : str
        Full path to the image file.

    Returns
    -------
    imagehash.ImageHash or None
        The perceptual hash, or None if the image cannot be opened.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        return imagehash.phash(img)
    except Exception:
        return None


def is_duplicate(img_path: str, seen_hashes: list, config) -> bool:
    """
    Check if an image is a near-duplicate of any previously accepted image.

    Compares the pHash of the new image against all hashes in seen_hashes.
    If the Hamming distance to any existing hash is within the threshold,
    the image is considered a duplicate and should be rejected.

    Parameters
    ----------
    img_path : str
        Full path to the candidate image.
    seen_hashes : list of imagehash.ImageHash
        Hashes of all previously accepted images (accumulated across the run).
    config : module
        The config module, used for HASH_DISTANCE_THRESHOLD.

    Returns
    -------
    bool
        True if the image is a duplicate (should be skipped),
        False if it is unique enough to keep.
    """
    new_hash = compute_hash(img_path)

    # If hashing fails (corrupt file), treat as duplicate to be safe
    if new_hash is None:
        return True

    # Compare against every previously accepted image hash
    for existing_hash in seen_hashes:
        distance = new_hash - existing_hash   # Hamming distance between bit strings
        if distance <= config.HASH_DISTANCE_THRESHOLD:
            return True   # Close enough to be considered a duplicate

    # Unique image — add its hash to the seen list and accept it
    seen_hashes.append(new_hash)
    return False