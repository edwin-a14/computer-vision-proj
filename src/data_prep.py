import os
import csv
import random
import cv2
import xml.etree.ElementTree as ET

# only want stop signs right now
label_normalization = {
    'stop': 'stop',
}

# Increased from 64 to 128 for better detail preservation
CHIP_SIZE = 128

# Reads annotation xml files from kaggle, outputs csv with xml data flattened into rows
# with each row corresponding to an object in the scene image
def convert_kaggle_annotations_to_csv(annotations_dir_path, images_dir_path):
    rows = []

    with os.scandir(annotations_dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                filename_without_ext, _ = os.path.splitext(entry.name)
                img_path = os.path.join(images_dir_path, f"{filename_without_ext}.png") # road123.xml => road123 => road123.png

                tree = ET.parse(entry.path)
                root = tree.getroot()

                all_object = root.findall('object')

                for obj in all_object:
                    label = obj.findtext('name')

                    if label not in label_normalization:
                        continue

                    bndbox = obj.find('bndbox')

                    xmin = int(bndbox.findtext('xmin'))
                    ymin = int(bndbox.findtext('ymin'))
                    xmax = int(bndbox.findtext('xmax'))
                    ymax = int(bndbox.findtext('ymax'))

                    # img_path, xmin, ymin, xmax, ymax, label, scene_id, split
                    # split is train/val/test
                    # set scene_id to filename, split to "" for now
                    rows.append([img_path, xmin, ymin, xmax, ymax, label_normalization[label], filename_without_ext, ""])                    
    return rows

# Creates chip of an object (stop sign, traffic light, etc) by using the csv flattened annotations
# to crop the box around the object, resizes it to 64 x 64 for consistency
def create_chips(rows):
    base_output_path = os.path.join('data', 'processed', 'chips')
    counts = {}

    for img_path, xmin, ymin, xmax, ymax, label, scene_id, split in rows:
        img = cv2.imread(img_path)

        y_upper_bound, x_upper_bound = img.shape[:2]

        x0, y0 = max(0, xmin), max(0, ymin)
        x1, y1 = min(x_upper_bound - 1, xmax), min(y_upper_bound - 1, ymax)
        chip = img[y0 : y1, x0 : x1]

        if chip.size == 0:
            continue

        #chip = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        chip = cv2.resize(chip, (CHIP_SIZE, CHIP_SIZE), interpolation=cv2.INTER_AREA)

        counts[scene_id] = counts.get(scene_id, 0) + 1

        split_path = os.path.join(base_output_path, split)
        label_path = os.path.join(split_path, label)

        os.makedirs(split_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        out_path = os.path.join(label_path, f"{scene_id}_{counts[scene_id]:06d}.png")

        cv2.imwrite(out_path, chip)

# Divides csv data randomly:
# Training data: 70%
# Validation data: 15%
# Test data: 15%
def split_by_scene(rows, train=0.7, val=0.15, seed=1337):
    scenes = {}

    for r in rows:
        scene_id = r[6]

        if scene_id in scenes:
            scenes[r[6]] += 1
        else:
            scenes[r[6]] = 0

    scene_ids = list(scenes.keys())
    random.Random(seed).shuffle(scene_ids)

    n = len(scene_ids)
    n_train = int(n * train)
    n_val = int(n * val)
    
    train_ids = set(scene_ids[:n_train])
    val_ids = set(scene_ids[n_train:n_train+n_val])
    
    for r in rows:
        scene_id = r[6]
        if scene_id in train_ids:
            r[7] = "train"
        elif scene_id in val_ids:
            r[7] = "val"
        else:
            r[7] = "test"
            
    return rows

# Creates background chips i.e. chips that don't have the object we're trying to train to detect. 
# This is done by dividing the image into 4 sections around the original chip, and cropping chips
# Chips are then resized to 64 x 64 for consistency
def create_background_chips(rows):
    base_output_path = os.path.join('data', 'processed', 'chips')
    count = 1
    
    for img_path, xmin, ymin, xmax, ymax, _, _, split in rows:
        img = cv2.imread(img_path)
        y_upper_bound, x_upper_bound = img.shape[:2]

        chip_x0, chip_y0 = xmin, ymin
        chip_x1, chip_y1 = xmax, ymax

        sections = [ 
            # (x_lower, y_lower), (x_high, y_high)
            [ (0, 0), (chip_x0, y_upper_bound - 1) ], # left side of chip
            [ (chip_x1, 0), (x_upper_bound - 1, y_upper_bound - 1) ], # right side of chip
            [ (0, 0), (x_upper_bound - 1, chip_y0)  ], # bottom side of chip
            [ (0, chip_y1), (x_upper_bound - 1, y_upper_bound - 1) ], # top side of chip
        ]

        for sec in sections:
            x_dist = sec[1][0] - sec[0][0]
            y_dist = sec[1][1] - sec[0][1]

            if x_dist < 40 or y_dist < 40:
                continue

            width = random.randint(40, min(x_dist, CHIP_SIZE))
            height = random.randint(40, min(y_dist, CHIP_SIZE))

            bg_chip_x0 = random.randint(0, max(1, x_dist - width))
            bg_chip_y0 = random.randint(0, max(1, y_dist - height))

            bg_chip_x1, bg_chip_y1 = bg_chip_x0 + width, bg_chip_y0 + height

            bg_chip = cv2.cvtColor(img[bg_chip_y0 : bg_chip_y1, bg_chip_x0 : bg_chip_x1], cv2.COLOR_BGR2GRAY)
            bg_chip = cv2.resize(bg_chip, (CHIP_SIZE, CHIP_SIZE))

            split_path = os.path.join(base_output_path, split)
            bg_path = os.path.join(split_path, "bg")

            os.makedirs(split_path, exist_ok=True)
            os.makedirs(bg_path, exist_ok=True)

            output_path = os.path.join(bg_path, f"bg_{count:06d}.png")
            cv2.imwrite(output_path, bg_chip)

            count += 1




def prep_data():
    path_to_kaggle_roadsign = os.path.join("data", "raw", "kaggle_roadsign")

    rows = convert_kaggle_annotations_to_csv(
        os.path.join(path_to_kaggle_roadsign, 'annotations'), 
        os.path.join(path_to_kaggle_roadsign, 'images'), 
    )

    rows = split_by_scene(rows)

    annotations_path = os.path.join("data", "annotations")
    os.makedirs(annotations_path, exist_ok=True)

    with open(os.path.join(annotations_path, "labels.csv"), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(['img_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'scene_id', 'split'])
        csv_writer.writerows(rows)

    processed_path = os.path.join("data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    create_chips(rows)
    create_background_chips(rows)

def main():
    prep_data()

if __name__ == "__main__":
    main()