
"""
Color Shape Preparation Tool
===========================
Extracts color histograms from labeled chip images, clusters color variations by sign type,
and saves color signatures for later comparison in detection pipelines.

Features:
- Extracts 7 traffic-relevant color bins (Red, Yellow, Blue, Orange, White/Gray, Black, Other)
- Computes HSV histograms for each sign type
- Saves color signatures to JSON for efficient comparison
"""

import os
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

def validate_histogram_against_signature(
    chip_hist: np.ndarray,
    signature: Dict,
    primary_bins: int = 6,
    std_multiplier: float = 1.5,
    presence_epsilon: float = 0.01,
    require_ratio: bool = True,
) -> bool:
    """
    Validate a chip color histogram against a learned signature.

    - Checks per-bin composition for the first `primary_bins` (excluding 'other').
    - Uses log-space pairwise color ratios for significant colors when available.
    - Significant bins (mean - std > 0) enforce lower and upper bounds.
    - Insignificant bins only enforce upper bounds.

    Returns True if the histogram matches the signature; otherwise False.
    """
    try:
        mean_hist = np.array(signature['mean_histogram'], dtype=float)
        std_hist = np.array(signature['std_histogram'], dtype=float)

        n_bins = min(primary_bins, len(chip_hist), len(mean_hist), len(std_hist))

        for i in range(n_bins):
            mean_val = mean_hist[i]
            std_val = std_hist[i]

            lower_bound = mean_val - std_multiplier * std_val
            upper_bound = mean_val + std_multiplier * std_val

            if chip_hist[i] < lower_bound or chip_hist[i] > upper_bound:
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
                            
                            # Only check positive ratios (i > j), relax upper bound by 2x
                            if exp_mu > 0:
                                # Check if chip ratio is too low (stricter lower bound)
                                if chip_log_ratio < exp_mu - exp_sigma:
                                    return False
                                # Relax upper bound: allow up to 2x the std deviation above mean
                                if chip_log_ratio > exp_mu + 2.0 * exp_sigma:
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
    Returns a dict with keys for each color mask used in the histogram extraction.
    """
    red_mask = (((h >= 0) & (h <= 12)) | ((h >= 168) & (h <= 180))) & (s >= 90) & (v >= 50)
    orange_mask = ((h >= 10) & (h <= 20)) & (s >= 100) & (v >= 110) & (v <= 190) & (~red_mask)
    yellow_mask = ((h >= 20) & (h <= 40)) & (s >= 100) & (v > 80) & (~red_mask) & (~orange_mask)
    blue_mask = ((h >= 95) & (h <= 135)) & (s >= 100) & (v > 50) & (~red_mask) & (~orange_mask) & (~yellow_mask)
    green_mask = ((h >= 35) & (h <= 85)) & (s > 80) & (v > 50) & (~red_mask) & (~orange_mask) & (~yellow_mask) & (~blue_mask)
    saturated_primaries = (red_mask | yellow_mask | blue_mask | orange_mask) & (s > 40)
    white_mask = (((s <= 100) & (v >= 90)) | (v >= 230) | ((s <= 25) & (v >= 180)) | (l_hls >= 205)) & (~red_mask) & (~yellow_mask) & (~blue_mask) & (~orange_mask)
    r = img[:, :, 2]; g = img[:, :, 1]; b = img[:, :, 0]
    black_mask = ((((r < 60) & (g < 60) & (b < 60)) | (v < 65) | (((r < 82) & (g < 82) & (b < 82)) & (v < 72) & (s < 170))) & (~((v >= 72) & (s < 35))) & (~white_mask) & (~red_mask) & (~yellow_mask) & (~blue_mask) & (~orange_mask) & (~green_mask))
    
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
    """
    Extract a 7-bin normalized histogram of traffic-relevant color categories from an image.
    Bins: red, yellow, blue, orange, white_light, black, other.
    """
    if img is None or img.size == 0:
        return np.zeros(7, dtype=np.float32)
    # Histogram is computed on the provided image as-is (no WB applied here)

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
    masks = build_color_masks(h, s, v, l_hls, img)

    # Compute histogram bins
    histogram[0] = np.sum(masks['red_mask']) / total_pixels
    histogram[1] = np.sum(masks['yellow_mask']) / total_pixels
    histogram[2] = np.sum(masks['blue_mask']) / total_pixels
    histogram[3] = np.sum(masks['orange_mask']) / total_pixels
    histogram[4] = np.sum(masks['white_mask']) / total_pixels
    histogram[5] = np.sum(masks['black_mask']) / total_pixels

    specific_color_mask = masks['red_mask'] | masks['yellow_mask'] | masks['blue_mask'] | masks['orange_mask'] | masks['green_mask'] | masks['white_mask'] | masks['black_mask']
    other_mask = (~specific_color_mask)
    histogram[6] = np.sum(other_mask) / total_pixels
    return histogram


def extract_hsv_histogram(img, h_bins=32, s_bins=32, v_bins=32):
    """
    Extract a full HSV histogram from an image and return the normalized concatenated result.
    """
    if img is None or img.size == 0:
        return np.zeros(h_bins + s_bins + v_bins)
    
    # Histogram is computed on the provided image as-is (no WB applied here)

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


def get_color_mask_integrals(img):
    """
    Given a BGR image, compute all color masks and their integral images for fast region histogram extraction.
    Returns a dict of integral images for each color bin (red, yellow, blue, orange, white, black, green, other).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_hls, l_hls, s_hls = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    masks = build_color_masks(h, s, v, l_hls, img)
    integral_masks = {k: cv2.integral(masks[k].astype(np.uint8)) for k in ['red_mask','yellow_mask','blue_mask','orange_mask','white_mask','black_mask','green_mask']}
    specific_color_mask = masks['red_mask'] | masks['yellow_mask'] | masks['blue_mask'] | masks['orange_mask'] | masks['green_mask'] | masks['white_mask'] | masks['black_mask']
    other_mask = (~specific_color_mask)
    integral_masks['other_mask'] = cv2.integral(other_mask.astype(np.uint8))
    return integral_masks


def load_chip_images_by_type(chip_base_path, split='train'):
    """
    Load all chip images organized by sign type from disk for a given split.
    Returns a dict mapping sign_type -> list of image arrays.
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
    Returns a dict mapping sign_type -> signature statistics.
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
        # indices: 0=R,1=Y,2=B,3=O,4=W,5=K,6=other
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
    """Save color signatures to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(color_signatures, f, indent=2)
    
    logging.info(f"Color signatures saved to: {output_path}")


def load_color_signatures(json_path: str) -> Dict:
    """Load color signatures from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def collect_color_pixels(chips_list, color_name):
    """
    Extract all RGB pixels from chips that fall within a specific color category.
    Returns a list of RGB tuples for pixels in this color range.
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
    Generates a pie chart and (optionally) a 3D scatter plot for each sign type.
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
        # Ensure pie chart always shows the 7 primary bins (R,Y,B,O,W,K,other).
        # If the signature has fewer bins, pad with zeros; if more, take the first 7.
        desired_bins = 7
        mean_hist7 = np.zeros(desired_bins, dtype=float)
        copy_n = min(len(mean_hist), desired_bins)
        if copy_n > 0:
            mean_hist7[:copy_n] = mean_hist[:copy_n]
        # Normalize for display (avoid zero-sum for matplotlib.pie)
        total = float(mean_hist7.sum())
        if total <= 0.0:
            # fallback: tiny uniform distribution
            mean_hist7[:] = 1.0 / desired_bins
        else:
            mean_hist7 /= total
        num_samples = sig['num_samples']
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- Subplot 1: Pie Chart of Mean Color Distribution ---
        ax1.pie(mean_hist7, labels=color_names, autopct='%1.1f%%', startangle=90,
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
            primary_color_names = ['red', 'yellow', 'blue', 'orange', 'white_light', 'black']
            limit = min(len(primary_color_names), len(mean_ratios))
            for i in range(limit):
                color_name = primary_color_names[i]
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
    # Save signatures to computations
    output_json = os.path.join('computations', 'color_signatures.json')
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
    
    # Ensure computations directory exists and save to JSON
    try:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
    except Exception:
        pass
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
