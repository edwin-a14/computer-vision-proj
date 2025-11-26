import os
import shutil
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def mine_hard_negatives():
    labels_path = 'data/annotations/labels.csv'
    detections_dir = 'data/processed/found_chips_ensemble'
    train_bg_dir = 'data/processed/chips/train/bg'
    
    os.makedirs(train_bg_dir, exist_ok=True)
    
    # Identify images with ground truth stop signs
    gt_images = set()
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['img_path']
                img_name = os.path.basename(img_path)
                gt_images.add(img_name)
            
    for img_folder in os.listdir(detections_dir):
        folder_path = os.path.join(detections_dir, img_folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        is_clean_background = True
        for gt_img in gt_images:
            if gt_img.startswith(img_folder + '.'):
                is_clean_background = False
                break
        
        if is_clean_background:
            # All chips in this folder are False Positives / Hard Negatives
            for file in os.listdir(folder_path):
                if file.endswith('.png') and file != 'result.png':
                    src_path = os.path.join(folder_path, file)
                    
                    new_name = f"hard_neg_{img_folder}_{file}"
                    dst_path = os.path.join(train_bg_dir, new_name)
                    
                    shutil.copy2(src_path, dst_path)
                        
    logging.info(f"Saved to {train_bg_dir}")

if __name__ == "__main__":
    mine_hard_negatives()
