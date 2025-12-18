import joblib
import cv2, numpy as np
from sklearn.svm import SVC  # Changed from LinearSVC to SVC for RBF kernel
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog
import os

# Should prob extract this somewhere else
CHIP_IDENTIFIERS = ["stop", "bg"]

def extract_color_histogram_features(img):
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])  # Hue: 0-180
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])  # Saturation: 0-255
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])  # Value: 0-255
    
    # Normalize histograms
    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
    
    # Concatenate all histograms
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    
    return color_features

def extract_improved_features(img):
    # No white balance applied
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hog_features = hog(
        hsv_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        channel_axis=2,
        feature_vector=True
    )
    color_features = extract_color_histogram_features(img)
    combined_features = np.concatenate([hog_features, color_features])
    return combined_features

# Loads chip images by split, then gets improved features from each image
# Returns array of features, and an array of the corresponding chip labels for use with SVC
def load_images_by_split(split):
    base_split_path = os.path.join("data", "processed", "chips", split)
    features, chip_labels = [], []

    for chip in CHIP_IDENTIFIERS:
        base_chip_path = os.path.join(base_split_path, chip)

        with os.scandir(base_chip_path) as entries:
            for entry in entries:
                img = cv2.imread(entry.path)
                if img is None:
                    continue
                
                # Extract improved features (HOG + color histograms)
                combined_features = extract_improved_features(img)
                features.append(combined_features)
                chip_labels.append(chip)

    return np.array(features), np.array(chip_labels)


def main():
    train_features, train_chip_labels = load_images_by_split("train")
    val_features, val_chip_labels = load_images_by_split("val")
    test_features, test_chip_labels = load_images_by_split("test")

    # Train SVM with RBF kernel (non-linear classifier)
    # RBF kernel can capture more complex decision boundaries than linear
    clf = SVC(
        kernel='rbf',              # Radial Basis Function kernel
        C=1.0,                     # Regularization parameter
        gamma='scale',             # Kernel coefficient (auto-scaled)
        class_weight='balanced',   # Handle class imbalance
        probability=True,          # Enable probability estimates
        max_iter=-1,               # No iteration limit
        cache_size=500             # Increase cache for faster training
    )
    clf.fit(train_features, train_chip_labels)

    compare_arr = [
        ("Validation Chips Set", (val_features, val_chip_labels)), 
        ("Test Chips Set", (test_features, test_chip_labels))
    ]

    for title, (features, chip_labels) in compare_arr:
        predictions = clf.predict(features)
        print("\n===============================")
        print(f"{title.upper()}")
        print("===============================")
        print(classification_report(chip_labels, predictions, digits=3))
        print("Confusion matrix:\n", confusion_matrix(chip_labels, predictions))

    computations_path = os.path.join("computations")
    os.makedirs(computations_path, exist_ok=True)

    joblib.dump(clf, os.path.join(computations_path, "hog_svm_stop_and_bg.pkl"))

if __name__ == "__main__":
    main()