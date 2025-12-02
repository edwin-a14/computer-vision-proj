"""
Color Shape Preparation Tool
=============================
Extracts color histograms from labeled chip images, clusters color variations by sign type
using PCA, and saves color signatures for later comparison in detection pipelines.

Features:
- Extracts 8 primary hue channels (Red, Orange, Yellow, Green, Cyan, Blue, Magenta, White/Gray)
- Computes HSV histograms for each sign type
- Applies PCA to identify most varied color dimensions per sign type
- Saves clustered color signatures to JSON for efficient comparison
"""

import os
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def validate_histogram_against_signature(
    chip_hist: np.ndarray,
    signature: Dict,
    primary_bins: int = 6,
    std_multiplier: float = 1.5,
    presence_epsilon: float = 0.01,
    require_ratio: bool = True,
    black_std_multiplier: float = 0.85,  # Stricter check for black if significant
) -> bool:
    """
    Validate a chip color histogram against a learned signature.

    - Checks per-bin composition for the first `primary_bins` (excluding 'other').
    - Uses log-space pairwise color ratios for significant colors when available.
    - Applies stricter std multiplier to black bin if it's a significant color.

    Returns True if the histogram matches the signature; otherwise False.
    """
    try:
        mean_hist = np.array(signature['mean_histogram'], dtype=float)
        std_hist = np.array(signature['std_histogram'], dtype=float)

        n_bins = min(primary_bins, len(chip_hist), len(mean_hist), len(std_hist))

        # 1) Per-bin checks
        for i in range(n_bins):
            mean_val = mean_hist[i]
            std_val = std_hist[i]
            
            # Use stricter multiplier for black bin (index 5) even if not significant
            if i == 5:
                multiplier = black_std_multiplier
                # For black, always check deviation from mean
                if abs(chip_hist[i] - mean_val) > multiplier * std_val:
                    return False
            elif mean_val - std_val > 0:
                if abs(chip_hist[i] - mean_val) > std_multiplier * std_val:
                    return False
            else:
                if chip_hist[i] > mean_val + std_val:
                    return False

        # 2) Pairwise log-space ratio checks (optional)
        ratio_means = signature.get('pairwise_ratio_means')
        ratio_stds = signature.get('pairwise_ratio_stds')
        ratio_evaluated = False
        if isinstance(ratio_means, dict) and isinstance(ratio_stds, dict):
            # Significant colors
            significant = [i for i in range(n_bins) if (mean_hist[i] - std_hist[i]) > 0]
            color_names = ['red', 'yellow', 'blue', 'orange', 'white_light', 'black']
            for i in significant:
                for j in significant:
                    if i == j:
                        continue
                    if chip_hist[i] > presence_epsilon and chip_hist[j] > presence_epsilon:
                        key = f"{color_names[i]}:{color_names[j]}"
                        if key in ratio_means and key in ratio_stds:
                            exp_mu = ratio_means.get(key)
                            exp_sigma = ratio_stds.get(key)
                            if exp_mu is None or exp_sigma is None:
                                continue
                            ratio_evaluated = True
                            chip_log_ratio = np.log(chip_hist[i] + 1e-10) - np.log(chip_hist[j] + 1e-10)
                            if abs(chip_log_ratio - exp_mu) > exp_sigma:
                                return False
        # If ratio checks are required but none were evaluated, reject
        if require_ratio and not ratio_evaluated:
            return False
        return True
    except Exception:
        # Be permissive on failure to avoid over-rejection
        return True
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Map HSV hue ranges to 7 color categories for traffic signs
# Hue in HSV: 0-180 in OpenCV (0-360 degrees / 2)
# Bins: red, yellow, blue, orange, white_light, black, other
COLOR_RANGES = {
    'red': [(0, 12), (168, 180)],        # Red wraps around (stop signs)
    'yellow': [(20, 40)],                # Yellow (speed limit, warning signs)
    'blue': [(95, 135)],                 # Blue (informational signs)
    'orange': [(10, 20)],                # Orange (construction/caution)
    'white_light': [(-1, -1)],           # White/Light (low saturation or high value)
    'black': [(-1, -1)],                 # Black (very dark, low value)
    'other': None,                       # Everything else not in above ranges
}


def build_color_masks(h, s, v, l_hls, img):
    """Build all color masks for the 7-bin histogram.
    
    Returns: dict with keys: red_mask, yellow_mask, blue_mask, orange_mask,
             white_mask, black_mask, green_mask, saturated_primaries
    """
    # Green mask: only overlaps with yellow (hue 35-40), exclude from yellow only
    green_mask = ((h >= 35) & (h <= 85)) & (s > 80) & (v > 50)
    
    # Primary color masks (only yellow needs green exclusion due to hue overlap at 35-40)
    red_mask = (((h >= 0) & (h <= 12)) | ((h >= 168) & (h <= 180))) & (s > 40) & (v > 80)
    yellow_mask = ((h >= 20) & (h <= 40)) & (s > 40) & (v > 80) & (~green_mask)
    blue_mask = ((h >= 95) & (h <= 135)) & (s > 40) & (v > 80)
    orange_mask = ((h >= 10) & (h <= 20)) & (s > 40) & (v > 80)
    
    # Exclusion masks
    purple_mask = (h >= 135) & (h <= 145)
    magenta_mask = ((h >= 145) & (h <= 170)) & (s > 40)
    green_from_white_mask = ((h >= 35) & (h <= 95))
    
    # Apply purple exclusion to blue
    blue_mask = blue_mask & (~purple_mask)
    
    # White criteria - multiple conditions
    grayscale_mask = (s < 55) & (v > 150)
    bright_mask = v > 220
    lightness_mask = l_hls > 215
    light_mask = (s < 65) & (v > 155)
    very_light_color_mask = (s < 85) & (v > 195)
    medium_light_gray_mask = (s < 50) & (v > 135)
    white_criteria = grayscale_mask | bright_mask | lightness_mask | light_mask | very_light_color_mask | medium_light_gray_mask
    
    # Exclude saturated primaries and green from white
    saturated_primaries = (red_mask | yellow_mask | blue_mask | orange_mask) & (s > 40)
    exclude_from_white = saturated_primaries | magenta_mask | green_from_white_mask
    white_mask = white_criteria & (~exclude_from_white)
    
    # Add very light desaturated pixels to white
    other_mask = ~(saturated_primaries | white_mask)
    lightest_other_mask = other_mask & (v > 200) & (s < 60) & (~green_from_white_mask)
    white_mask = white_mask | lightest_other_mask
    
    # Black mask
    rgb_b = img[:, :, 0]
    rgb_g = img[:, :, 1]
    rgb_r = img[:, :, 2]
    max_rgb = np.maximum.reduce([rgb_r, rgb_g, rgb_b])
    strict_low_triplet = (rgb_r < 60) & (rgb_g < 60) & (rgb_b < 60)
    very_dark_value = v < 65
    expanded_dark = (max_rgb < 82) & (v < 72) & (s < 170)
    medium_gray_exclusion = (v >= 72) & (s < 35)
    black_mask = (strict_low_triplet | very_dark_value | expanded_dark) & (~medium_gray_exclusion) & (~white_mask) & (~saturated_primaries)
    
    return {
        'red_mask': red_mask,
        'yellow_mask': yellow_mask,
        'blue_mask': blue_mask,
        'orange_mask': orange_mask,
        'white_mask': white_mask,
        'black_mask': black_mask,
        'green_mask': green_mask,
        'saturated_primaries': saturated_primaries
    }


def extract_color_histogram(img):
    """Extract histogram of 7 traffic-relevant color categories.

    Categories (ordered):
      0 red          : saturated red hues (stop sign body)
      1 yellow       : saturated yellow hues (warning signs)
      2 blue         : saturated blue hues (info/regulatory)
      3 orange       : saturated orange slice (construction/caution)
      4 white_light  : light / reflective / gray (letters, borders)
      5 black        : dark text / symbols (low value)
      6 other        : everything else (greens, background, excluded hues)

    Separating black from 'other' reduces noise and enables later symbol analysis.
    Returns a 7-bin normalized histogram.
    """
    if img is None or img.size == 0:
        return np.zeros(7, dtype=np.float32)
    
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Also compute HLS to get lightness (helps detect light gray/white)
        h_hls, l_hls, s_hls = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    except Exception:
        return np.zeros(5)
    height, width = h.shape
    
    histogram = np.zeros(7, dtype=np.float32)
    total_pixels = float(height * width)
    if total_pixels <= 0:
        return np.ones(7, dtype=np.float32) / 7.0

    # Build all color masks using helper function
    masks = build_color_masks(h, s, v, h_hls, img)
    
    # Compute histogram bins (green already excluded from primaries)
    histogram[0] = np.sum(masks['red_mask']) / total_pixels
    histogram[1] = np.sum(masks['yellow_mask']) / total_pixels
    histogram[2] = np.sum(masks['blue_mask']) / total_pixels
    histogram[3] = np.sum(masks['orange_mask']) / total_pixels
    histogram[4] = np.sum(masks['white_mask']) / total_pixels
    histogram[5] = np.sum(masks['black_mask']) / total_pixels
    
    specific_color_mask = masks['saturated_primaries'] | masks['white_mask'] | masks['black_mask']
    other_mask = (~specific_color_mask) | masks['green_mask']
    histogram[6] = np.sum(other_mask) / total_pixels
    return histogram


def extract_hsv_histogram(img, h_bins=32, s_bins=32, v_bins=32):
    """
    Extract full HSV histogram from an image.
    
    Args:
        img: Input image (BGR)
        h_bins, s_bins, v_bins: Number of bins for each HSV channel
    
    Returns:
        Concatenated HSV histogram, normalized
    """
    if img is None or img.size == 0:
        return np.zeros(h_bins + s_bins + v_bins)
    
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        return np.zeros(h_bins + s_bins + v_bins)
    
    hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
    
    # Normalize
    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
    
    return np.concatenate([hist_h, hist_s, hist_v])


def load_chip_images_by_type(chip_base_path, split='train'):
    """
    Load all chip images organized by sign type from disk.
    
    Args:
        chip_base_path: Base path to chips (e.g., 'data/processed/chips')
        split: Which split to load ('train', 'val', 'test')
    
    Returns:
        Dict mapping sign_type -> list of image arrays
    """
    chips_by_type = {}
    split_path = os.path.join(chip_base_path, split)
    
    if not os.path.exists(split_path):
        logging.warning(f"Split path not found: {split_path}")
        return chips_by_type
    
    # Scan for sign type directories
    for sign_type in os.listdir(split_path):
        type_path = os.path.join(split_path, sign_type)
        
        if not os.path.isdir(type_path):
            continue
        
        chips_by_type[sign_type] = []
        
        # Load all images in this type directory
        for img_file in os.listdir(type_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(type_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None and img.size > 0:
                chips_by_type[sign_type].append(img)
        
        logging.info(f"  Loaded {len(chips_by_type[sign_type])} chips for type '{sign_type}'")
    
    return chips_by_type


def extract_color_signatures(chips_by_type: Dict[str, List[np.ndarray]], 
                             use_primary_hues=True):
    """
    Extract color histograms from chips for each sign type.
    
    Args:
        chips_by_type: Dict mapping sign_type -> list of chip images
        use_primary_hues: If True, use 5 traffic sign colors; else use full HSV
    
    Returns:
        Dict mapping sign_type -> {'num_samples': int, 'mean_histogram': array, 'std_histogram': array, 'histogram_dtype': str}
    """
    color_signatures = {}
    
    for sign_type, chips in chips_by_type.items():
        # Skip 'bg' label
        if sign_type.lower() == 'bg':
            logging.info(f"Skipping histogram calculation for '{sign_type}' (background)")
            continue
        
        if not chips:
            logging.warning(f"No chips found for sign type '{sign_type}'")
            continue
        
        logging.info(f"Processing color signatures for '{sign_type}' ({len(chips)} chips)...")
        
        # Extract histograms for all chips of this type
        histograms = []
        for chip in chips:
            if use_primary_hues:
                hist = extract_color_histogram(chip)
            else:
                hist = extract_hsv_histogram(chip)
            histograms.append(hist)

        histograms = np.array(histograms)

        # Compute mean and std of color signatures
        mean_hist = np.mean(histograms, axis=0)
        # For each color bin, calculate stddev across all chips
        # histograms shape: (num_chips, num_bins)
        std_hist = np.std(histograms, axis=0)

        # Compute pairwise log-space color ratios (include black, exclude 'other')
        # New indices: 0=R,1=Y,2=B,3=O,4=W,5=K,6=other
        if use_primary_hues and histograms.shape[1] == 7:
            primary_indices = [0, 1, 2, 3, 4, 5]
            primary_hists = histograms[:, primary_indices]
            primary_names = ['red', 'yellow', 'blue', 'orange', 'white_light', 'black']
            
            # Compute all pairwise ratios using log-space for better stability
            pairwise_ratios_means = {}
            pairwise_ratios_stds = {}
            
            for i, name_i in enumerate(primary_names):
                for j, name_j in enumerate(primary_names):
                    if i != j:
                        # Only compute ratio if both colors have significant presence on average
                        if primary_hists[:, i].mean() > 0.05 and primary_hists[:, j].mean() > 0.05:
                            # Filter chips where both colors are present (> 1%)
                            valid_chips = (primary_hists[:, i] > 0.01) & (primary_hists[:, j] > 0.01)
                            
                            if valid_chips.sum() > 10:  # Need at least 10 valid chips
                                # Compute log-space ratios for stability
                                log_ratio = np.log(primary_hists[valid_chips, i] + 1e-10) - \
                                           np.log(primary_hists[valid_chips, j] + 1e-10)
                                
                                pairwise_ratios_means[f'{name_i}:{name_j}'] = float(np.mean(log_ratio))
                                pairwise_ratios_stds[f'{name_i}:{name_j}'] = float(np.std(log_ratio))
                            else:
                                # Not enough valid chips, use fallback
                                pairwise_ratios_means[f'{name_i}:{name_j}'] = None
                                pairwise_ratios_stds[f'{name_i}:{name_j}'] = None
                        else:
                            # At least one color not significant, skip ratio
                            pairwise_ratios_means[f'{name_i}:{name_j}'] = None
                            pairwise_ratios_stds[f'{name_i}:{name_j}'] = None
        else:
            pairwise_ratios_means = None
            pairwise_ratios_stds = None

        # Logical abbreviated bin names for histogram_dtype
        bin_names = ['R', 'Y', 'B', 'O', 'W', 'K', 'X']
        histogram_dtype = '_'.join(bin_names)

        logging.info(f"  Mean histogram shape: {mean_hist.shape}, Std per bin: {std_hist}")

        # Compute normalized color ratios (primary colors only, excluding 'other')
        if use_primary_hues and histograms.shape[1] == 7:
            # Extract primary color histograms (exclude 'other')
            primary_hists = histograms[:, :6]
            
            # Compute mean ratios
            mean_ratios = np.mean(primary_hists, axis=0)  # Mean of each primary color
            std_ratios = np.std(primary_hists, axis=0)   # Std dev of each primary color
            
            # Store in signature
            color_signatures[sign_type] = {
                'num_samples': len(chips),
                'mean_histogram': mean_hist.tolist(),
                'std_histogram': std_hist.tolist(),
                'pairwise_ratio_means': pairwise_ratios_means,
                'pairwise_ratio_stds': pairwise_ratios_stds,
                'histogram_dtype': histogram_dtype if use_primary_hues else 'hsv_32_32_32',
                'histogram_dim': histograms.shape[1],
                'mean_ratios': mean_ratios.tolist(),
                'std_ratios': std_ratios.tolist(),
            }
        else:
            color_signatures[sign_type] = {
                'num_samples': len(chips),
                'mean_histogram': mean_hist.tolist(),
                'std_histogram': std_hist.tolist(),
                'pairwise_ratio_means': pairwise_ratios_means,
                'pairwise_ratio_stds': pairwise_ratios_stds,
                'histogram_dtype': histogram_dtype if use_primary_hues else 'hsv_32_32_32',
                'histogram_dim': histograms.shape[1],
            }
    
    return color_signatures


def save_color_signatures(color_signatures: Dict, output_path: str):
    """Save color signatures to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(color_signatures, f, indent=2)
    
    logging.info(f"Color signatures saved to: {output_path}")


def load_color_signatures(json_path: str) -> Dict:
    """Load color signatures from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def collect_color_pixels(chips_list, color_name):
    """
    Extract all RGB pixels from chips that fall within a specific color category.
    
    Uses HSV color space with saturation and value filters to exclude black/gray/dark pixels
    from primary colors (red, yellow, blue) while capturing light grays and whites.
    
    Args:
        chips_list: List of chip images (BGR)
        color_name: Color category ('red', 'yellow', 'blue', 'white_light', 'other')
    
    Returns:
        List of RGB tuples for pixels in this color range
    """
    color_pixels = []

    for idx, chip_img in enumerate(chips_list):
        if chip_img is None or chip_img.size == 0:
            continue

        # Convert BGR to HSV and HLS for detection
        try:
            hsv = cv2.cvtColor(chip_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
        except Exception:
            continue

        try:
            _, l_hls, _ = cv2.split(cv2.cvtColor(chip_img, cv2.COLOR_BGR2HLS))
        except Exception:
            l_hls = np.zeros_like(h)

        # Build all color masks using helper function
        masks = build_color_masks(h, s, v, l_hls, chip_img)
        
        # Select appropriate mask for the requested color (green already excluded from primaries)
        if color_name == 'red':
            color_match = masks['red_mask']
        elif color_name == 'yellow':
            color_match = masks['yellow_mask']
        elif color_name == 'blue':
            color_match = masks['blue_mask']
        elif color_name == 'orange':
            color_match = masks['orange_mask']
        elif color_name == 'white_light':
            color_match = masks['white_mask']
        elif color_name == 'black':
            color_match = masks['black_mask']
        elif color_name == 'other':
            specific_colors = masks['saturated_primaries'] | masks['white_mask'] | masks['black_mask']
            color_match = (~specific_colors) | masks['green_mask']
        else:
            continue

        # Apply mask (full chip)
        pixel_mask = color_match
        if np.sum(pixel_mask) == 0:
            continue

        b_pixels = chip_img[:, :, 0][pixel_mask]
        g_pixels = chip_img[:, :, 1][pixel_mask]
        r_pixels = chip_img[:, :, 2][pixel_mask]

        num_pixels = len(r_pixels)
        if num_pixels > 500:
            indices = np.random.choice(num_pixels, 500, replace=False)
            r_pixels = r_pixels[indices]
            g_pixels = g_pixels[indices]
            b_pixels = b_pixels[indices]

        for r, g, b in zip(r_pixels, g_pixels, b_pixels):
            color_pixels.append((float(r), float(g), float(b)))

    return color_pixels


def visualize_color_signatures(color_signatures: Dict, output_dir: str, chips_by_type: Dict[str, List] = None):
    """
    Create visual representations of color signatures for each sign type.
    
    Generates:
    1. Pie chart showing mean color distribution for the sign type
    2. 3D scatter plot of actual chip pixel colors by color category
    
    Args:
        color_signatures: Dict mapping sign_type -> signature data
        output_dir: Directory to save visualization files
        chips_by_type: Dict mapping sign_type -> list of chip images (for 3D scatter)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 7-bin color names and RGB values for visualization
    color_names = ['red', 'yellow', 'blue', 'orange', 'white_light', 'black', 'other']
    color_rgb = {
        'red': (255, 0, 0),
        'yellow': (255, 255, 0),
        'blue': (0, 0, 255),
        'orange': (255, 165, 0),
        'white_light': (220, 220, 220),
        'black': (30, 30, 30),
        'other': (128, 128, 128)
    }
    
    for sign_type, sig in color_signatures.items():
        # Skip 'bg' label
        if sign_type.lower() == 'bg':
            logging.info(f"Skipping visualization for '{sign_type}' (background)")
            continue
        
        logging.info(f"Creating visualizations for '{sign_type}'...")
        
        mean_hist = np.array(sig['mean_histogram'])
        num_samples = sig['num_samples']
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- Subplot 1: Pie Chart of Mean Color Distribution ---
        ax1.pie(mean_hist, labels=color_names, autopct='%1.1f%%', startangle=90,
            colors=[np.array(color_rgb[cn]) / 255.0 for cn in color_names],
            wedgeprops=dict(edgecolor='w'))
        ax1.set_title(f'{sign_type.upper()} - Mean Color Distribution\n({num_samples} samples)', 
                 fontsize=14, fontweight='bold')

        # --- Subplot 2: Color Composition Info ---
        ax2.axis('off')

        # Create text summary
        summary_text = f"Sign Type: {sign_type.upper()}\n\n"
        summary_text += f"Number of Samples: {num_samples}\n\n"
        summary_text += "Mean Color Composition (mean ± std):\n"
        summary_text += "-" * 40 + "\n"
        if sig.get('mean_ratios') is not None:
            mean_ratios = np.array(sig['mean_ratios'])
            std_ratios = np.array(sig['std_ratios'])
            primary_color_names = ['red', 'yellow', 'blue', 'orange', 'white_light']
            for i, color_name in enumerate(primary_color_names):
                ratio_val = mean_ratios[i] * 100
                std_val = std_ratios[i] * 100
                display_name = color_name.replace('_', '/')
                summary_text += f"{display_name:15s}: {ratio_val:.1f}%±{std_val:.1f}%\n"
        summary_text += "-" * 40 + "\n"

        # Add pairwise color ratios
        if sig.get('pairwise_ratio_means') is not None:
            summary_text += "\n" + "=" * 40 + "\n"
            summary_text += "Top 3 Most Consistent Color Ratios:\n"
            summary_text += "(Log-space: positive = more of first color,\n"
            summary_text += " negative = more of second color)\n"
            summary_text += "-" * 40 + "\n"
            # Find top 3 ratios with smallest deviation (excluding None values)
            ratio_stds = [(k, v) for k, v in sig['pairwise_ratio_stds'].items() if v is not None]
            if len(ratio_stds) > 0:
                ratio_stds_sorted = sorted(ratio_stds, key=lambda x: x[1])[:3]
                for ratio_key, std_val in ratio_stds_sorted:
                    mean_val = sig['pairwise_ratio_means'][ratio_key]
                    if mean_val is not None:
                        display_key = ratio_key.replace('_light', '')
                        summary_text += f"{display_key:15s}: {mean_val:.3f}±{std_val:.3f}\n"
        # Show normalized color ratios (primary colors only, excluding 'other')
        # ...existing code...

        ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save pie chart figure (adjust layout to avoid label overlap)
        output_file = os.path.join(output_dir, f'{sign_type}_color_signature.png')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"  Saved pie chart to: {output_file}")

        # Create 3D scatter plot if we have chip data
        if chips_by_type is not None and sign_type in chips_by_type:
            logging.info(f"  Creating 3D scatter plot for '{sign_type}'...")
            from mpl_toolkits.mplot3d import Axes3D
            chips = chips_by_type[sign_type]
            chips_with_white = sum(1 for c in chips if extract_color_histogram(c)[4] > 0.01)
            logging.info(f"    {sign_type}: chips with white bin >1%: {chips_with_white} / {len(chips)}")

            import math
            cols = 4 if len(color_names) > 6 else 3
            rows = math.ceil(len(color_names) / cols)
            fig = plt.figure(figsize=(5.5 * cols, 5.0 * rows))
            pixel_counts = {}
            max_total_points = 5000  # max points per category to plot
            for idx, color_name in enumerate(color_names):
                ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
                pixels = collect_color_pixels(chips_by_type[sign_type], color_name)
                pixel_counts[color_name] = len(pixels)
                logging.info(f"    {sign_type}: {color_name} pixels collected: {pixel_counts[color_name]}")

                if len(pixels) > 0:
                    pixels = np.array(pixels)
                    if len(pixels) > max_total_points:
                        sel = np.random.choice(len(pixels), max_total_points, replace=False)
                        pixels = pixels[sel]

                    r_vals = pixels[:, 0]
                    g_vals = pixels[:, 1]
                    b_vals = pixels[:, 2]
                    colors_normalized = pixels / 255.0;
                    ax.scatter(r_vals, g_vals, b_vals, c=colors_normalized, s=30, alpha=0.5, edgecolors='none')
                    ax.set_xlabel('Red', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Green', fontsize=11, fontweight='bold')
                    ax.set_zlabel('Blue', fontsize=11, fontweight='bold')
                    ax.xaxis.set_tick_params(labelsize=8)
                    ax.yaxis.set_tick_params(labelsize=8)
                    ax.zaxis.set_tick_params(labelsize=8)
                    ax.set_xlim(0, 255)
                    ax.set_ylim(0, 255)
                    ax.set_zlim(0, 255)
                    ax.set_xticks([0, 64, 128, 192, 255])
                    ax.set_yticks([0, 64, 128, 192, 255])
                    ax.set_zticks([0, 64, 128, 192, 255])
                    display_name = color_name.replace('_', ' ').upper()
                    ax.set_title(f'{display_name}\n({pixel_counts[color_name]:,} pixels)', fontsize=12, fontweight='bold', pad=15)
                else:
                    ax.set_title(f'{color_name.replace("_", " ").upper()} (0 pixels)', fontsize=12, fontweight='bold', pad=15)
            plt.suptitle(f'{sign_type.upper()} - 3D RGB Color Distribution', fontsize=16, fontweight='bold', y=0.98)
            plt.subplots_adjust(hspace=0.5, wspace=0.35, top=0.92, bottom=0.08)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            scatter_file = os.path.join(output_dir, f'{sign_type}_3d_scatter.png')
            plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"  Saved 3D scatter plot to: {scatter_file}")


def print_color_signature_summary(color_signatures: Dict):
    """Print summary of loaded color signatures, including black bin."""
    print("\n" + "="*60)
    print("COLOR SIGNATURE SUMMARY")
    print("="*60)

    # Updated color ordering matches histogram: red, yellow, blue, orange, white_light, black, other
    color_names = ['red', 'yellow', 'blue', 'orange', 'white_light', 'black', 'other']

    for sign_type, sig in color_signatures.items():
        if sign_type.lower() == 'bg':
            continue

        print(f"\nSign Type: '{sign_type}'")
        print(f"  Samples: {sig['num_samples']}")
        print(f"  Histogram Type: {sig['histogram_dtype']}")
        print(f"  Histogram Dimension: {sig['histogram_dim']}")

        mean_hist = np.array(sig['mean_histogram'])
        std_hist = np.array(sig.get('std_histogram', [0]*len(mean_hist)))
        print("  Mean Color Distribution (primary bins + other):")
        for i, color_name in enumerate(color_names):
            if i < len(mean_hist):
                mean_pct = mean_hist[i] * 100
                std_pct = std_hist[i] * 100 if i < len(std_hist) else 0
                print(f"    {color_name:15s}: {mean_pct:6.2f}%  (± {std_pct:5.2f}%)")


def main():
    """Main pipeline: load chips, extract signatures, save to JSON, visualize."""
    
    # Configuration
    chip_base_path = os.path.join('data', 'processed', 'chips')
    split = 'train'  # Use training set for color signature extraction
    output_json = os.path.join('data', 'processed', 'color_signatures.json')
    # Updated graphs directory moved directly under data/graphs
    graphs_output_dir = os.path.join('data', 'graphs')
    
    use_primary_hues = True  # Use 5-color histogram instead of full HSV
    
    logging.info(f"Loading chip images from: {chip_base_path}")
    logging.info(f"Using split: '{split}'")
    
    # Load chip images by type
    chips_by_type = load_chip_images_by_type(chip_base_path, split=split)
    
    if not chips_by_type:
        logging.error("No chips loaded! Check your data path.")
        return
    
    logging.info(f"Found {len(chips_by_type)} sign types")
    
    # Extract color signatures
    color_signatures = extract_color_signatures(
        chips_by_type, 
        use_primary_hues=use_primary_hues
    )
    
    # Save to JSON
    save_color_signatures(color_signatures, output_json)
    
    # Print summary
    print_color_signature_summary(color_signatures)
    
    # Create visualizations (pass chips_by_type for 3D scatter plots)
    logging.info(f"Creating visualizations in: {graphs_output_dir}")
    visualize_color_signatures(color_signatures, graphs_output_dir, chips_by_type=chips_by_type)
    
    print(f"\n✓ Color signature extraction and visualization complete!")
    print(f"  Signatures: {output_json}")
    print(f"  Graphs: {graphs_output_dir}")


if __name__ == "__main__":
    main()
