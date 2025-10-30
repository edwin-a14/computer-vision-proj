import joblib
import cv2, numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog
import os

# Should prob extract this somewhere else
CHIP_IDENTIFIERS = ["stop", "bg"]

# Loads chip images by split, then gets HOG from each image
# Returns array of HOGs, and an array of the corresponding chip labels for use with LinearSVC
def load_images_by_split(split):
    base_split_path = os.path.join("data", "processed", "chips", split)
    hogs, chip_labels = [], []

    for chip in CHIP_IDENTIFIERS:
        base_chip_path = os.path.join(base_split_path, chip)

        with os.scandir(base_chip_path) as entries:
            for entry in entries:
                img = cv2.imread(entry.path)
                # Can use color or grayscale, channel axis specifies the axis of color channels
                # Using L1 norm for block normalization instead of the default L2-Hys
                hogs.append(hog(img, channel_axis=2, block_norm='L1'))
                chip_labels.append(chip)

    return np.array(hogs), np.array(chip_labels)


def main():
    train_hogs, train_chip_labels = load_images_by_split("train")
    val_hogs, val_chip_labels = load_images_by_split("val")
    test_hogs, test_chip_labels = load_images_by_split("test")

    # Trains LinearSVC (Linear SVM implementation) using HOGs + Chip labels
    clf = LinearSVC(class_weight = "balanced", max_iter = 20000)
    clf.fit(train_hogs, train_chip_labels)

    compare_arr = [ ("Validation Chips Set", (val_hogs, val_chip_labels)), ("Test Chips Set", (test_hogs, test_chip_labels))]

    for title, (hogs, chip_labels) in compare_arr:
        predictions = clf.predict(hogs)
        print("===============================")
        print(f"{title.upper()}")
        print("===============================")
        print(classification_report(chip_labels, predictions, digits = 3))
        print("Confusion matrix:\n", confusion_matrix(chip_labels, predictions))

    computations_path = os.path.join("computations")
    os.makedirs(computations_path, exist_ok=True)

    joblib.dump(clf, os.path.join(computations_path, "hog_svm_stop_and_bg.pkl"))

if __name__ == "__main__":
    main()