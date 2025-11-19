
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil

def get_augmentation_pipeline():

    return A.Compose([
        # Geometric transformations
        A.OneOf([
            A.Rotate(limit=30, p=1.0),
            A.Affine(rotate=(-30, 30), shear=(-10, 10), p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=30, p=1.0),
        ], p=0.8),
        
        A.OneOf([
            A.Perspective(scale=(0.05, 0.15), p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.5),
        
        A.HorizontalFlip(p=0.3),
        
        # Color and lighting variations (crucial for outdoor signs)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomGamma(gamma_limit=(50, 150), p=1.0),
        ], p=0.9),
        
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=40, p=1.0),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.7),
        
        # Weather and environmental conditions
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),
        
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
            A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=1.0),
        ], p=0.5),
        
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=1.0),
        ], p=0.3),
        
        # Occlusion and quality degradation
        A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, p=1.0),
            A.GridDropout(ratio=0.3, random_offset=True, p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.ImageCompression(quality_lower=60, quality_upper=90, p=1.0),
        ], p=0.3),
        
        # Normalize last
        A.ToGray(p=0.1),  # Occasional grayscale
    ])

def augment_dataset(input_dir, output_dir, augmentations_per_image=15, target_size=128):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    for class_name in ['stop', 'bg']:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    transform = get_augmentation_pipeline()
    
    stats = {'stop': 0, 'bg': 0}
    
    for class_name in ['stop', 'bg']:
        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name
        
        if not class_input_dir.exists():
            print(f"Warning: {class_input_dir} does not exist")
            continue
        
        image_files = list(class_input_dir.glob('*.jpg')) + list(class_input_dir.glob('*.png'))
        
        
        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            # Read original image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Resize to target size
            img = cv2.resize(img, (target_size, target_size))
            
            # Save original
            original_output = class_output_dir / f"{img_path.stem}_orig{img_path.suffix}"
            cv2.imwrite(str(original_output), img)
            stats[class_name] += 1
            
            # Generate augmented versions
            for i in range(augmentations_per_image):
                try:
                    augmented = transform(image=img)['image']
                    
                    # Save augmented image
                    aug_output = class_output_dir / f"{img_path.stem}_aug{i:03d}{img_path.suffix}"
                    cv2.imwrite(str(aug_output), augmented)
                    stats[class_name] += 1
                    
                except Exception as e:
                    print(f"Error augmenting {img_path.name}: {e}")
                    continue
    
    return stats

def balance_classes(dataset_dir, target_ratio=1.0):

    dataset_dir = Path(dataset_dir)
    
    stop_files = list((dataset_dir / 'stop').glob('*'))
    bg_files = list((dataset_dir / 'bg').glob('*'))
    
    stop_count = len(stop_files)
    bg_count = len(bg_files)

    target_stop_count = int(bg_count * target_ratio)
    
    if target_stop_count > stop_count:
        # Need to oversample stop signs
        needed = target_stop_count - stop_count
        
        transform = get_augmentation_pipeline()
        
        for i in tqdm(range(needed), desc="Oversampling"):
            # Randomly select a stop sign image
            src_img_path = np.random.choice(stop_files)
            img = cv2.imread(str(src_img_path))
            
            if img is not None:
                augmented = transform(image=img)['image']
                
                output_path = dataset_dir / 'stop' / f"{src_img_path.stem}_balance{i:04d}{src_img_path.suffix}"
                cv2.imwrite(str(output_path), augmented)


def create_augmented_dataset(base_dir='data/processed/chips', 
                             output_dir='data/processed/chips_augmented',
                             augmentations_per_image=15,
                             balance_ratio=0.8):

    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    
    train_stats = augment_dataset(
        input_dir=base_dir / 'train',
        output_dir=output_dir / 'train',
        augmentations_per_image=augmentations_per_image
    )
    
    balance_classes(output_dir / 'train', target_ratio=balance_ratio)
    
    val_stats = augment_dataset(
        input_dir=base_dir / 'val',
        output_dir=output_dir / 'val',
        augmentations_per_image=5  # Fewer augmentations for val
    )
    
    test_output_dir = output_dir / 'test'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in ['stop', 'bg']:
        src_dir = base_dir / 'test' / class_name
        dst_dir = test_output_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        if src_dir.exists():
            for img_file in src_dir.glob('*'):
                shutil.copy2(img_file, dst_dir / img_file.name)
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            stop_count = len(list((split_dir / 'stop').glob('*')))
            bg_count = len(list((split_dir / 'bg').glob('*')))
            total = stop_count + bg_count
            ratio = bg_count / stop_count if stop_count > 0 else 0
            
            print(f"\n{split.upper()}:")
            print(f"  Stop: {stop_count}")
            print(f"  Background: {bg_count}")
            print(f"  Total: {total}")
            print(f"  Ratio: 1:{ratio:.2f}")
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment training data for improved model performance')
    parser.add_argument('--input-dir', type=str, default='data/processed/chips',
                       help='Input directory with original dataset')
    parser.add_argument('--output-dir', type=str, default='data/processed/chips_augmented',
                       help='Output directory for augmented dataset')
    parser.add_argument('--augmentations', type=int, default=15,
                       help='Number of augmented versions per image')
    parser.add_argument('--balance-ratio', type=float, default=0.8,
                       help='Target stop:bg ratio (0.8 = 1:1.25)')
    
    args = parser.parse_args()
    
    create_augmented_dataset(
        base_dir=args.input_dir,
        output_dir=args.output_dir,
        augmentations_per_image=args.augmentations,
        balance_ratio=args.balance_ratio
    )
