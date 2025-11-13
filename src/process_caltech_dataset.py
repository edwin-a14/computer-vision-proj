import cv2
import numpy as np
from pathlib import Path
import random
import shutil
from tqdm import tqdm


def parse_caltech_annotations(annotation_file):
    annotations = {}
    
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 2:
                continue
            
            # Extract image number from annotation filename
            mat_file = parts[0].strip()
            img_num = mat_file.replace('annotation_', '').replace('.mat', '')
            img_filename = f"image_{img_num}.jpg"
            
            # Parse bounding box: xmin ymin xmax ymax
            coords = parts[1].strip().split()
            if len(coords) == 4:
                xmin, ymin, xmax, ymax = map(int, coords)
                annotations[img_filename] = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'width': xmax - xmin,
                    'height': ymax - ymin
                }
    
    return annotations


def extract_stop_sign_chip(img, bbox, target_size=128, padding_ratio=0.1):
    """Extract and resize stop sign chip with padding"""
    xmin, ymin, width, height = bbox['xmin'], bbox['ymin'], bbox['width'], bbox['height']
    
    # Add padding
    pad_w = int(width * padding_ratio)
    pad_h = int(height * padding_ratio)
    
    x1 = max(0, xmin - pad_w)
    y1 = max(0, ymin - pad_h)
    x2 = min(img.shape[1], xmin + width + pad_w)
    y2 = min(img.shape[0], ymin + height + pad_h)
    
    # Extract chip
    chip = img[y1:y2, x1:x2]
    
    if chip.size == 0 or chip.shape[0] < 10 or chip.shape[1] < 10:
        return None
    
    # Resize while maintaining aspect ratio
    h, w = chip.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w < 10 or new_h < 10:
        return None
    
    resized = cv2.resize(chip, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def extract_background_chips(img, bbox, num_chips=3, target_size=128):
    """Extract random background chips from image (avoiding stop sign region)."""
    h, w = img.shape[:2]
    
    # Stop sign region
    sign_xmin, sign_ymin = bbox['xmin'], bbox['ymin']
    sign_xmax = sign_xmin + bbox['width']
    sign_ymax = sign_ymin + bbox['height']
    
    chips = []
    attempts = 0
    max_attempts = 20
    
    while len(chips) < num_chips and attempts < max_attempts:
        attempts += 1
        
        # Random size between 50-200 pixels
        size = random.randint(50, min(200, h, w))
        
        # Random position
        x = random.randint(0, max(0, w - size))
        y = random.randint(0, max(0, h - size))
        
        # Check if overlaps with stop sign (IoU < 0.1)
        overlap_x = max(0, min(x + size, sign_xmax) - max(x, sign_xmin))
        overlap_y = max(0, min(y + size, sign_ymax) - max(y, sign_ymin))
        overlap_area = overlap_x * overlap_y
        
        chip_area = size * size
        sign_area = bbox['width'] * bbox['height']
        union_area = chip_area + sign_area - overlap_area
        
        iou = overlap_area / union_area if union_area > 0 else 0
        
        # Accept if minimal overlap with stop sign
        if iou < 0.1:
            chip = img[y:y+size, x:x+size]
            
            if chip.size > 0 and chip.shape[0] > 10 and chip.shape[1] > 10:
                # Resize to target size
                resized = cv2.resize(chip, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                chips.append(resized)
    
    return chips


def process_caltech_dataset(caltech_dir='data/raw/CALTECH101_STOP_SIGN',
                            output_dir='data/processed/chips',
                            target_size=128,
                            train_ratio=0.7,
                            val_ratio=0.15):

    caltech_dir = Path(caltech_dir)
    output_dir = Path(output_dir)
    
    images_dir = caltech_dir / 'stop_sign'
    annotations_file = caltech_dir / 'stop_sign_annotations_converted.txt'
    
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        return
    
    if not annotations_file.exists():
        print(f"Error: {annotations_file} does not exist")
        return
    
    annotations = parse_caltech_annotations(annotations_file)
    
    # Get all image files
    image_files = sorted(list(images_dir.glob('image_*.jpg')))
    
    # Split into train/val/test
    random.seed(42)
    random.shuffle(image_files)
    
    n_train = int(len(image_files) * train_ratio)
    n_val = int(len(image_files) * val_ratio)
    
    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train+n_val]
    test_images = image_files[n_train+n_val:]
    
    # Process each split
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, split_images in splits.items():        
        stop_dir = output_dir / split_name / 'stop'
        bg_dir = output_dir / split_name / 'bg'
        
        stop_dir.mkdir(parents=True, exist_ok=True)
        bg_dir.mkdir(parents=True, exist_ok=True)
        
        # Count existing chips to avoid overwriting
        existing_stop = len(list(stop_dir.glob('*.jpg')))
        existing_bg = len(list(bg_dir.glob('*.jpg')))
        
        stop_counter = existing_stop
        bg_counter = existing_bg
        
        for img_file in tqdm(split_images, desc=f"  Extracting {split_name} chips"):
            img_name = img_file.name
            
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Get annotation
            if img_name not in annotations:
                print(f"   Warning: No annotation for {img_name}")
                continue
            
            bbox = annotations[img_name]
            
            # Extract stop sign chip
            stop_chip = extract_stop_sign_chip(img, bbox, target_size=target_size)
            if stop_chip is not None:
                output_path = stop_dir / f"caltech_{stop_counter:04d}.jpg"
                cv2.imwrite(str(output_path), stop_chip)
                stop_counter += 1
            
            # Extract background chips
            bg_chips = extract_background_chips(img, bbox, num_chips=3, target_size=target_size)
            for bg_chip in bg_chips:
                output_path = bg_dir / f"caltech_bg_{bg_counter:04d}.jpg"
                cv2.imwrite(str(output_path), bg_chip)
                bg_counter += 1



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Caltech101 stop sign dataset')
    parser.add_argument('--caltech-dir', type=str, 
                       default='data/raw/CALTECH101_STOP_SIGN',
                       help='Directory containing Caltech stop sign images')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/chips',
                       help='Output directory for processed chips')
    parser.add_argument('--target-size', type=int, default=128,
                       help='Target size for chips (default: 128)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    process_caltech_dataset(
        caltech_dir=args.caltech_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
