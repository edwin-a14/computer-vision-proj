import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_detection_stats(stats_path: str) -> Dict:
    """Load detection statistics from JSON file."""
    with open(stats_path, 'r') as f:
        return json.load(f)


def create_side_by_side_comparison(image_name: str, hog_dir: str, cnn_dir: str, output_dir: str):
    hog_result_path = os.path.join(hog_dir, image_name, 'result.png')
    cnn_result_path = os.path.join(cnn_dir, image_name, 'result.png')
    
    if not os.path.exists(hog_result_path) or not os.path.exists(cnn_result_path):
        logging.warning(f"Missing results for {image_name}")
        return False
    
    hog_img = cv2.imread(hog_result_path)
    cnn_img = cv2.imread(cnn_result_path)
    
    if hog_img is None or cnn_img is None:
        logging.warning(f"Failed to load images for {image_name}")
        return False
    
    # Convert BGR to RGB for matplotlib
    hog_img = cv2.cvtColor(hog_img, cv2.COLOR_BGR2RGB)
    cnn_img = cv2.cvtColor(cnn_img, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(hog_img)
    axes[0].set_title('HOG-SVM Classifier', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cnn_img)
    axes[1].set_title('CNN Classifier', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(f'Detection Comparison: {image_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{image_name}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def generate_statistical_comparison(hog_stats: Dict, cnn_stats: Dict, output_dir: str):
    logging.info("\n" + "="*60)
    logging.info("CLASSIFIER COMPARISON STATISTICS")
    logging.info("="*60)
    
    # Overall statistics
    logging.info("\nOverall Statistics:")
    logging.info(f"  Total images processed: {hog_stats['total_images']}")
    
    logging.info("\n  HOG-SVM:")
    logging.info(f"    Total detections: {hog_stats['total_detections']}")
    logging.info(f"    Images with detections: {hog_stats['images_with_detections']}")
    logging.info(f"    Detection rate: {hog_stats['images_with_detections']/hog_stats['total_images']*100:.2f}%")
    logging.info(f"    Avg detections per image (overall): {hog_stats['total_detections']/hog_stats['total_images']:.2f}")
    if hog_stats['images_with_detections'] > 0:
        logging.info(f"    Avg detections per image (with detections): {hog_stats['total_detections']/hog_stats['images_with_detections']:.2f}")
    
    logging.info("\n  CNN:")
    logging.info(f"    Total detections: {cnn_stats['total_detections']}")
    logging.info(f"    Images with detections: {cnn_stats['images_with_detections']}")
    logging.info(f"    Detection rate: {cnn_stats['images_with_detections']/cnn_stats['total_images']*100:.2f}%")
    logging.info(f"    Avg detections per image (overall): {cnn_stats['total_detections']/cnn_stats['total_images']:.2f}")
    if cnn_stats['images_with_detections'] > 0:
        logging.info(f"    Avg detections per image (with detections): {cnn_stats['total_detections']/cnn_stats['images_with_detections']:.2f}")
    
    hog_detections = {img['name']: img['detections'] for img in hog_stats['images_processed']}
    cnn_detections = {img['name']: img['detections'] for img in cnn_stats['images_processed']}
    
    both_detected = 0
    only_hog = 0
    only_cnn = 0
    neither = 0
    agreement = 0
    
    for img_name in hog_detections:
        h = hog_detections.get(img_name, 0)
        c = cnn_detections.get(img_name, 0)
        
        if h == c:
            agreement += 1
        
        if h > 0 and c > 0:
            both_detected += 1
        elif h > 0 and c == 0:
            only_hog += 1
        elif h == 0 and c > 0:
            only_cnn += 1
        else:
            neither += 1
    
    logging.info("\n  Agreement Analysis:")
    logging.info(f"    Both detected: {both_detected} images")
    logging.info(f"    Only HOG-SVM detected: {only_hog} images")
    logging.info(f"    Only CNN detected: {only_cnn} images")
    logging.info(f"    Neither detected: {neither} images")
    logging.info(f"    Exact count agreement: {agreement}/{hog_stats['total_images']} ({agreement/hog_stats['total_images']*100:.2f}%)")
    
    create_comparison_visualizations(hog_stats, cnn_stats, output_dir)
    
    comparison_report = {
        'hog_svm': {
            'total_detections': hog_stats['total_detections'],
            'images_with_detections': hog_stats['images_with_detections'],
            'detection_rate': hog_stats['images_with_detections']/hog_stats['total_images']
        },
        'cnn': {
            'total_detections': cnn_stats['total_detections'],
            'images_with_detections': cnn_stats['images_with_detections'],
            'detection_rate': cnn_stats['images_with_detections']/cnn_stats['total_images']
        },
        'agreement': {
            'both_detected': both_detected,
            'only_hog': only_hog,
            'only_cnn': only_cnn,
            'neither': neither,
            'exact_agreement': agreement,
            'exact_agreement_rate': agreement/hog_stats['total_images']
        }
    }
    
    report_path = os.path.join(output_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    logging.info(f"\nDetailed report saved to: {report_path}")


def create_comparison_visualizations(hog_stats: Dict, cnn_stats: Dict, output_dir: str):
    
    hog_detections = {img['name']: img['detections'] for img in hog_stats['images_processed']}
    cnn_detections = {img['name']: img['detections'] for img in cnn_stats['images_processed']}
    
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    classifiers = ['HOG-SVM', 'CNN']
    totals = [hog_stats['total_detections'], cnn_stats['total_detections']]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(classifiers, totals, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Detections', fontsize=12)
    ax1.set_title('Total Detections Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    images_with = [hog_stats['images_with_detections'], cnn_stats['images_with_detections']]
    bars = ax2.bar(classifiers, images_with, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Images with Detections', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3 = plt.subplot(2, 3, 3)
    rates = [
        hog_stats['images_with_detections']/hog_stats['total_images']*100,
        cnn_stats['images_with_detections']/cnn_stats['total_images']*100
    ]
    bars = ax3.bar(classifiers, rates, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Detection Rate (%)', fontsize=12)
    ax3.set_title('Detection Rate Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4 = plt.subplot(2, 3, 4)
    hog_counts = [hog_detections.get(img['name'], 0) for img in hog_stats['images_processed']]
    cnn_counts = [cnn_detections.get(img['name'], 0) for img in hog_stats['images_processed']]
    ax4.scatter(hog_counts, cnn_counts, alpha=0.5, s=50)
    max_val = max(max(hog_counts), max(cnn_counts)) + 1
    ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect agreement')
    ax4.set_xlabel('HOG-SVM Detections', fontsize=12)
    ax4.set_ylabel('CNN Detections', fontsize=12)
    ax4.set_title('Per-Image Detection Agreement', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    bins = np.arange(0, max(max(hog_counts), max(cnn_counts)) + 2) - 0.5
    ax5.hist([hog_counts, cnn_counts], bins=bins, label=['HOG-SVM', 'CNN'], 
             alpha=0.6, color=colors, edgecolor='black')
    ax5.set_xlabel('Number of Detections per Image', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Distribution of Detection Counts', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    both_detected = sum(1 for h, c in zip(hog_counts, cnn_counts) if h > 0 and c > 0)
    only_hog = sum(1 for h, c in zip(hog_counts, cnn_counts) if h > 0 and c == 0)
    only_cnn = sum(1 for h, c in zip(hog_counts, cnn_counts) if h == 0 and c > 0)
    neither = sum(1 for h, c in zip(hog_counts, cnn_counts) if h == 0 and c == 0)
    
    categories = ['Both', 'Only HOG', 'Only CNN', 'Neither']
    values = [both_detected, only_hog, only_cnn, neither]
    colors_cat = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    
    wedges, texts, autotexts = ax6.pie(values, labels=categories, autopct='%1.1f%%',
                                        colors=colors_cat, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax6.set_title('Detection Agreement Categories', fontsize=12, fontweight='bold')
    
    plt.suptitle('Classifier Comparison: HOG-SVM vs CNN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'comparison_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Visualization saved to: {viz_path}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare classifier results')
    parser.add_argument('--hog-dir', type=str, default='data/processed/found_chips_hog',
                        help='Directory containing HOG-SVM results')
    parser.add_argument('--cnn-dir', type=str, default='data/processed/found_chips_cnn',
                        help='Directory containing CNN results')
    parser.add_argument('--output-dir', type=str, default='data/processed/classifier_comparison',
                        help='Directory to save comparison results')
    parser.add_argument('--max-comparisons', type=int, default=100,
                        help='Maximum number of side-by-side comparisons to generate (default: 100)')
    parser.add_argument('--skip-images', action='store_true',
                        help='Skip generating side-by-side image comparisons (only generate stats)')
    parser.add_argument('--prioritize-detections', action='store_true', default=True,
                        help='Prioritize showing images where at least one model detected something (default: True)')
    
    args = parser.parse_args()
    
    logging.info("Starting classifier comparison...")
    
    if not os.path.exists(args.hog_dir):
        logging.error(f"HOG-SVM results directory not found: {args.hog_dir}")
        return
    
    if not os.path.exists(args.cnn_dir):
        logging.error(f"CNN results directory not found: {args.cnn_dir}")
        return
    
    hog_stats_path = os.path.join(args.hog_dir, 'detection_stats_hog.json')
    cnn_stats_path = os.path.join(args.cnn_dir, 'detection_stats_cnn.json')
    
    if not os.path.exists(hog_stats_path):
        logging.error(f"HOG-SVM statistics not found: {hog_stats_path}")
        return
    
    if not os.path.exists(cnn_stats_path):
        logging.error(f"CNN statistics not found: {cnn_stats_path}")
        return
    
    hog_stats = load_detection_stats(hog_stats_path)
    cnn_stats = load_detection_stats(cnn_stats_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    generate_statistical_comparison(hog_stats, cnn_stats, args.output_dir)
    
    if not args.skip_images:
        logging.info(f"\nGenerating side-by-side comparisons (max {args.max_comparisons})...")
        
        # Create detection lookup for prioritization
        hog_detections_dict = {img['name']: img['detections'] for img in hog_stats['images_processed']}
        cnn_detections_dict = {img['name']: img['detections'] for img in cnn_stats['images_processed']}
        
        # Separate images into categories
        images_with_detections = []
        images_without_detections = []
        
        for item in hog_stats['images_processed']:
            img_name = item['name']
            hog_count = hog_detections_dict.get(img_name, 0)
            cnn_count = cnn_detections_dict.get(img_name, 0)
            
            if hog_count > 0 or cnn_count > 0:
                # Prioritize images with more detections and disagreement
                priority = max(hog_count, cnn_count) + abs(hog_count - cnn_count)
                images_with_detections.append((img_name, priority, hog_count, cnn_count))
            else:
                images_without_detections.append((img_name, 0, 0, 0))
        
        # Sort images with detections by priority (descending)
        images_with_detections.sort(key=lambda x: x[1], reverse=True)
        
        if args.prioritize_detections:
            # Prioritize images with detections
            selected_images = images_with_detections[:args.max_comparisons]
            
            # If we have room, add some without detections for completeness
            remaining = args.max_comparisons - len(selected_images)
            if remaining > 0:
                selected_images.extend(images_without_detections[:remaining])
            
            logging.info(f"  Prioritizing images with detections:")
            logging.info(f"    - {len([x for x in selected_images if x[2] > 0 or x[3] > 0])} with detections")
            logging.info(f"    - {len([x for x in selected_images if x[2] == 0 and x[3] == 0])} without detections")
        else:
            all_images = images_with_detections + images_without_detections
            selected_images = all_images[:args.max_comparisons]
        
        successful = 0
        for i, (img_name, priority, hog_count, cnn_count) in enumerate(selected_images):
            if i % 10 == 0:
                logging.info(f"  Processing {i+1}/{len(selected_images)}")
            
            image_name = os.path.splitext(img_name)[0]
            if create_side_by_side_comparison(image_name, args.hog_dir, args.cnn_dir, args.output_dir):
                successful += 1
        
        logging.info(f"\nGenerated {successful} side-by-side comparisons")
        logging.info(f"  Images with at least one detection: {len(images_with_detections)}")
        logging.info(f"  Images with no detections: {len(images_without_detections)}")
    
    logging.info(f"\nComparison complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
