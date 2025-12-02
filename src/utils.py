import os
import cv2
from typing import Optional


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
        cv2.putText(image, f"{lab}:{val:.2f}", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 12
    return image


def chip_path(directory: str, index: int, file_name: str, label: Optional[str] = None):
    parts = os.path.splitext(file_name)
    label_suffix = f"_{label}" if label else ""
    return os.path.join(directory, parts[0] + '_' + str(index) + label_suffix + parts[-1])


def remove_previous_outputs(image_dir: str, base_name: str, start_index: int = 1):
    if not os.path.exists(image_dir):
        return
    try:
        for file in os.listdir(image_dir):
            if file == 'result.png':
                continue
            if file.endswith('.png') and file.startswith(base_name + '_'):
                try:
                    rest = file[len(base_name) + 1:]
                    idx_str = rest.split('_')[0]
                    idx = int(idx_str) if idx_str.isdigit() else start_index
                except Exception:
                    idx = start_index
                if idx >= start_index:
                    os.remove(os.path.join(image_dir, file))
        for sub in ['candidates', 'validated']:
            subdir = os.path.join(image_dir, sub)
            if os.path.exists(subdir):
                for f in os.listdir(subdir):
                    fp = os.path.join(subdir, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
    except Exception:
        pass