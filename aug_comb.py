import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.core.bbox_utils import convert_bboxes_to_albumentations
import copy

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.RandomSizedBBoxSafeCrop(width=320, height=320, p=0.5, min_coverage=0.5),
    A.Resize(320, 320)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3))

labels_to_augment = ['Other plastic bottle', 'Clear plastic bottle', 'Glass bottle', 
                     'Plastic bottle cap', 'Metal bottle cap', 'Drink can', 'Other carton', 
                     'Corrugated carton', 'Disposable plastic cup', 'Other plastic', 'Normal paper', 
                     'Plastic straw', 'Styrofoam piece']

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_info = []
    filename = root.find('filename').text
    image_path = root.find('path').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        image_info.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return filename, image_path, image_info, (width, height), tree

def augment_data(image_path, image_info):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    bboxes = [obj['bbox'] for obj in image_info]
    category_ids = [obj['name'] for obj in image_info]

    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    augmented_categories = transformed['category_ids']

    image_height, image_width, _ = augmented_image.shape
    valid_bboxes = []
    valid_categories = []

    for idx, bbox in enumerate(augmented_bboxes):
        x_min, y_min, x_max, y_max = bbox
        if 0 <= x_min < image_width and 0 <= y_min < image_height and \
           0 < x_max <= image_width and 0 < y_max <= image_height and \
           x_max > x_min and y_max > y_min:  # Ensure bbox is valid
            valid_bboxes.append(bbox)
            valid_categories.append(augmented_categories[idx])

    augmented_image_info = []
    for idx, bbox in enumerate(valid_bboxes):
        augmented_image_info.append({
            'name': valid_categories[idx],
            'bbox': bbox
        })

    if len(augmented_image_info) == 0:
        print(f"No valid bounding boxes left after augmentation for image {image_path}")
        return None, None

    return augmented_image, augmented_image_info

def save_augmented_data(tree, image_info, augmented_image, output_image_path, output_xml_path, new_image_name):
    cv2.imwrite(output_image_path, augmented_image)

    root = tree.getroot()
    root.find('filename').text = new_image_name
    root.find('path').text = output_image_path

    for idx, obj in enumerate(root.findall('object')):
        if idx >= len(image_info):
            root.remove(obj)
            continue

        bbox = image_info[idx]['bbox']
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(int(bbox[0]))
        bndbox.find('ymin').text = str(int(bbox[1]))
        bndbox.find('xmax').text = str(int(bbox[2]))
        bndbox.find('ymax').text = str(int(bbox[3]))
        obj.find('name').text = image_info[idx]['name']

    # Remove extra objects in case they were not augmented
    while len(root.findall('object')) > len(image_info):
        root.remove(root.findall('object')[-1])

    # Save the updated XML file
    tree.write(output_xml_path)

def process_and_augment_dataset(annotations_dir, augmented_images_dir, augmented_xml_dir, augment_count=5):
    error_log_path = "error_log.txt"

    if not os.path.exists(augmented_images_dir):
        os.makedirs(augmented_images_dir)
    if not os.path.exists(augmented_xml_dir):
        os.makedirs(augmented_xml_dir)

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    index = 0
    with open(error_log_path, 'w') as error_log:
        for xml_file in xml_files:
            xml_path = os.path.join(annotations_dir, xml_file)
            try:
                filename, image_path, image_info, _, tree = parse_xml(xml_path)
                print(f"Processing {image_path}")
            except Exception as e:
                error_message = f"Error parsing XML file {xml_file}: {e}\n"
                print(error_message.strip())
                error_log.write(error_message)
                continue

            # Check if all labels in the image are in the list of labels to augment
            labels_in_image = [obj['name'] for obj in image_info]
            if any(label in labels_to_augment for label in labels_in_image):
                # Apply augmentations
                for _ in range(augment_count):
                    try:
                        augmented_image, augmented_image_info = augment_data(image_path, image_info)

                        if augmented_image is not None:
                            new_image_name = f"{index}.jpg"
                            output_image_path = os.path.join(augmented_images_dir, new_image_name)
                            output_xml_path = os.path.join(augmented_xml_dir, f"{index}.xml")

                            save_augmented_data(copy.deepcopy(tree), augmented_image_info, augmented_image, output_image_path, output_xml_path, new_image_name)
                            index += 1
                            print(f"Processed and saved: {output_image_path}, {output_xml_path}")
                        else:
                            print(f"Skipping augmentation for image {image_path} due to invalid data.")
                    except Exception as e:
                        error_message = f"Error augmenting {image_path} for XML file {xml_file}: {e}\n"
                        print(error_message.strip())
                        error_log.write(error_message)
                        continue
            else:
                # Save the original image without augmentation
                new_image_name = f"{index}.jpg"
                output_image_path = os.path.join(augmented_images_dir, new_image_name)
                output_xml_path = os.path.join(augmented_xml_dir, f"{index}.xml")
                cv2.imwrite(output_image_path, cv2.imread(image_path))
                tree.find('filename').text = new_image_name
                tree.find('path').text = output_image_path
                tree.write(output_xml_path)
                index += 1
                print(f"Saved original image: {output_image_path}, {output_xml_path}")

# Paths to directories
annotations_dir = "data13_320/train/"
augmented_images_dir = "TACO/data13_320/augmented/"
augmented_xml_dir = "data13_320/augmented/"

process_and_augment_dataset(annotations_dir, augmented_images_dir, augmented_xml_dir)
