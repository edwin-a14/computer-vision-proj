import os
import cv2
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import logging

logger = logging.getLogger(__name__)


def apply_gray_world(img_bgr, saturation_threshold: float = 0.5):
    """Apply a simple per-channel Gray-World white balance (environment-safe)."""
    if img_bgr is None or not hasattr(img_bgr, 'shape'):
        return img_bgr
    try:
        img = img_bgr.astype(np.float32)
        if img.size == 0 or img.ndim != 3 or img.shape[2] != 3:
            return img_bgr
        # Compute per-channel means and target gray
        means = img.reshape(-1, 3).mean(axis=0)
        gray = float(means.mean())
        eps = 1e-6
        gains = gray / (means + eps)
        # Optional clamp to avoid extreme amplification
        gains = np.clip(gains, 0.5, 2.5)
        balanced = img * gains.reshape(1, 1, 3)
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        return balanced
    except Exception:
        return img_bgr

def draw_histogram_overlay(image, chip_hist, x: int, y: int, w: int, color=(0, 255, 0)):
    if chip_hist is None or len(chip_hist) < 6:
        return image
    vals = [float(chip_hist[i]) for i in range(6)]
    labels = ['R', 'Y', 'B', 'O', 'W', 'K']
    font_scale = 0.35
    thickness = 1
    x_offset = x + w + 5
    y_offset = y + 12
    for lab, val in zip(labels, vals):
        cv2.putText(image, f"{lab}:{val:.2f}".upper(), (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 12
    return image

def save_and_load_all_masks_for_debug(debug_dir, base_prefix):
    """
    Load all relevant debug masks (hsv, lab, hsv_wb, lab_wb) for visualization.
    Returns: hsv_mask, lab_mask, hsv_mask_wb, lab_mask_wb (all as grayscale or None)
    """
    import cv2
    import os
    def load_mask(path):
        if os.path.exists(path):
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None
    hsv_mask = load_mask(os.path.join(debug_dir, f"{base_prefix}_hsv_mask_1.png"))
    lab_mask = load_mask(os.path.join(debug_dir, f"{base_prefix}_lab_mask_1.png"))
    hsv_mask_wb = load_mask(os.path.join(debug_dir, f"{base_prefix}_hsv_mask_wb_1.png"))
    lab_mask_wb = load_mask(os.path.join(debug_dir, f"{base_prefix}_lab_mask_wb_1.png"))
    return hsv_mask, lab_mask, hsv_mask_wb, lab_mask_wb

def save_debug_mask_visualization(debug_dir, base_prefix, final_img, hsv_mask, lab_mask, hsv_mask_wb, lab_mask_wb):
    """Save a debug visualization of masks and result image."""
    def to_rgb(mask):
        if mask is None:
            img = np.ones((final_img.shape[0], final_img.shape[1], 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'MISSING MASK'
            font_scale = 0.8
            thickness = 2
            color = (180, 180, 180)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (img.shape[1] - tw) // 2
            y = (img.shape[0] + th) // 2
            cv2.putText(img, text.upper(), (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            return img
        if len(mask.shape) == 2:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask
    h, w = final_img.shape[:2]
    def resize_to_h(im):
        return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_NEAREST) if im.shape[0] != h else im
    hsv = resize_to_h(to_rgb(hsv_mask))
    hsv_wb = resize_to_h(to_rgb(hsv_mask_wb))
    lab = resize_to_h(to_rgb(lab_mask))
    lab_wb = resize_to_h(to_rgb(lab_mask_wb))
    imgs = [hsv, hsv_wb, final_img, lab, lab_wb]
    minw = min(im.shape[1] for im in imgs)
    imgs = [cv2.resize(im, (minw, h), interpolation=cv2.INTER_NEAREST) if im.shape[1] != minw else im for im in imgs]
    concat = np.concatenate(imgs, axis=1)
    debug_vis_path = os.path.join(debug_dir, f"{base_prefix}_mask_debug.png")
    cv2.imwrite(debug_vis_path, concat)


def chip_path(directory: str, index: int, file_name: str, label: Optional[str] = None):
    parts = os.path.splitext(file_name)
    label_suffix = f"_{label}" if label else ""
    return os.path.join(directory, parts[0] + '_' + str(index) + label_suffix + parts[-1])

def save_chip(output_dir: str, base_name: str, index: int, image, label: Optional[str] = None, prefix: Optional[str] = None):
    try:
        os.makedirs(output_dir, exist_ok=True)
        name = base_name
        if prefix:
            name = f"{prefix}"
        parts = os.path.splitext(base_name if base_name else 'chip.png')
        ext = parts[-1] if len(parts) > 1 and parts[-1] else '.png'
        label_suffix = f"_{label}" if label else ""
        filename = f"{(os.path.splitext(base_name)[0] if base_name else 'chip')}_{index}{label_suffix}{ext}" if not prefix else f"{prefix}_{index}{ext}"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, image)
        return path
    except Exception:
        return None


def remove_previous_outputs(image_dir: str):
    """
    Recursively remove all files and subdirectories in the given directory except 'result.png'.
    Cleans up all previous outputs before writing new ones.
    """
    import shutil
    if not os.path.exists(image_dir):
        return
    try:
        for file in os.listdir(image_dir):
            fp = os.path.join(image_dir, file)
            if file == 'result.png':
                continue
            if os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
    except Exception:
        pass


def _safe_imread(path: str, to_rgb: bool = False):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if to_rgb and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _make_montage(images, tile_h: int, tile_w: int, cols: int = 5):
    if not images:
        return None
    tiles = []
    for im in images:
        if im is None:
            continue
        try:
            resized = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tiles.append(resized)
        except Exception:
            continue
    if not tiles:
        return None
    cols = max(1, cols)
    rows = (len(tiles) + cols - 1) // cols
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    row_imgs = []
    for r in range(rows):
        row = tiles[r*cols:(r+1)*cols]
        row_imgs.append(np.hstack(row))
    return np.vstack(row_imgs)

def create_compact_visualization_for_image(image_base: str, classifier_dir: str, max_candidates: int = 12, log: bool = True) -> Optional[str]:
    """Create a compact diagram using Matplotlib grid layout:
    - Top row: HSV, LAB, Combined masks with labels (keeps aspect ratio).
    - Middle: Annotated result image centered (keeps aspect ratio).
    - Bottom: Candidate chips in a small grid, preserving aspect (no squashing).
    Saves into the classifier/image subdirectory and returns output path, or None if nothing to visualize.
    """
    base_dir = os.path.join(classifier_dir, image_base)
    if not os.path.exists(base_dir):
        return None
    if log:
        logger.info(f"[compact] {classifier_dir}/{image_base} -> start")

    debug_dir = os.path.join(base_dir, 'debug_masks')
    cand_dir = os.path.join(base_dir, 'candidates')
    result_path = os.path.join(base_dir, 'result.png')

    # Load mask versions: non-wb (no postfix) and wb (with _wb)
    hsv_path = os.path.join(debug_dir, f"{image_base}_hsv_mask_1.png")
    lab_path = os.path.join(debug_dir, f"{image_base}_lab_mask_1.png")
    comb_path = os.path.join(debug_dir, f"{image_base}_combined_mask_1.png")
    wb_hsv_path = os.path.join(debug_dir, f"{image_base}_hsv_mask_wb_1.png")
    wb_lab_path = os.path.join(debug_dir, f"{image_base}_lab_mask_wb_1.png")
    wb_comb_path = os.path.join(debug_dir, f"{image_base}_combined_mask_wb_1.png")

    hsv = _safe_imread(hsv_path)
    lab = _safe_imread(lab_path)
    comb = _safe_imread(comb_path)
    wb_hsv = _safe_imread(wb_hsv_path)
    wb_lab = _safe_imread(wb_lab_path)
    wb_comb = _safe_imread(wb_comb_path)

    result_img = _safe_imread(result_path)
    result_rgb = None
    if result_img is not None:
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    if log:
        logger.info(f"[compact] masks: hsv={'y' if hsv is not None else 'n'}, lab={'y' if lab is not None else 'n'}, comb={'y' if comb is not None else 'n'}; wb masks: hsv={'y' if wb_hsv is not None else 'n'}, lab={'y' if wb_lab is not None else 'n'}, comb={'y' if wb_comb is not None else 'n'}")

    has_any = any([hsv is not None, lab is not None, comb is not None, wb_hsv is not None, wb_lab is not None, wb_comb is not None]) or os.path.exists(cand_dir) or result_rgb is not None
    if not has_any:
        return None

    # Convert masks to RGB for plotting
    def to_rgb(im):
        if im is None:
            # Return a white image with 'MISSING MASK' watermark for missing masks
            img = np.ones((128, 128, 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'MISSING MASK'
            font_scale = 0.42
            thickness = 1
            color = (180, 180, 180)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (img.shape[1] - tw) // 2
            y = (img.shape[0] + th) // 2
            cv2.putText(img, text.upper(), (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            return img
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Build figure using GridSpec to avoid squashing
    fig = plt.figure(figsize=(13, 8), dpi=150)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.7], width_ratios=[1, 1, 1], hspace=0.15, wspace=0.08)

    # Top row: HSV, LAB, Combined (non-wb)
    for i, (im, title) in enumerate([
        (hsv, 'HSV'),
        (lab, 'LAB'),
        (comb, 'Combined')
    ]):
        ax = fig.add_subplot(gs[0, i])
        ax.axis('off')
        ax.imshow(to_rgb(im))
        ax.set_title(title, fontsize=10)

    # Middle row: WB HSV (left), result (center), WB LAB (right)
    ax_mid_left = fig.add_subplot(gs[1, 0])
    ax_mid_left.axis('off')
    ax_mid_left.imshow(to_rgb(wb_hsv))
    ax_mid_left.set_title('WB HSV', fontsize=10)

    ax_mid_center = fig.add_subplot(gs[1, 1])
    ax_mid_center.axis('off')
    if result_rgb is not None:
        ax_mid_center.imshow(result_rgb)
    ax_mid_center.set_title('Annotated Result', fontsize=10)

    ax_mid_right = fig.add_subplot(gs[1, 2])
    ax_mid_right.axis('off')
    ax_mid_right.imshow(to_rgb(wb_lab))
    ax_mid_right.set_title('WB LAB', fontsize=10)

    # Bottom row: candidate chips grid (unchanged)
    chip_cols = 4
    groups = []
    for subname in ['hsv', 'lab', 'combined']:
        subdir = os.path.join(cand_dir, subname)
        grp = []
        if os.path.exists(subdir):
            files = [f for f in os.listdir(subdir) if f.endswith('.png')]
            files.sort()
            for f in files[:min(len(files), chip_cols)]:
                im = _safe_imread(os.path.join(subdir, f))
                if im is None:
                    continue
                grp.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        groups.append(grp)
    if log:
        logger.info(f"[compact] chips: hsv={len(groups[0])}, lab={len(groups[1])}, comb={len(groups[2])}")
    tile_h, tile_w = 64, 64
    for col_idx, group in enumerate(groups):
        ax = fig.add_subplot(gs[2, col_idx])
        ax.axis('off')
        if group:
            imgs = group[:chip_cols]
            tiles = []
            for img in imgs:
                try:
                    h, w = img.shape[:2]
                    scale = min(tile_w / w, tile_h / h)
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    tile = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255
                    y0 = (tile_h - new_h) // 2
                    x0 = (tile_w - new_w) // 2
                    tile[y0:y0+new_h, x0:x0+new_w] = thumb
                    tiles.append(tile)
                except Exception:
                    continue
            if tiles:
                while len(tiles) < chip_cols:
                    tiles.append(np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255)
                spacer_w = 6
                spacer = np.ones((tile_h, spacer_w, 3), dtype=np.uint8) * 255
                row_parts = []
                for k, t in enumerate(tiles[:chip_cols]):
                    row_parts.append(t)
                    if k < chip_cols - 1:
                        row_parts.append(spacer)
                mosaic = np.hstack(row_parts)
                ax.imshow(mosaic, interpolation='nearest')

    out_path = os.path.join(base_dir, f"{image_base}_compact.png")
    try:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        if log:
            logger.info(f"[compact] saved -> {out_path}")
        return out_path
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        if log:
            logger.warning(f"[compact] save failed -> {out_path}")
        return None