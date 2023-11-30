import settings
import os
import cv2
import pybboxes as pbx

COCO_LABELS_PATH = settings.raw_labels_path
IMAGES_DIR = settings.images_dir
YOLO_LABELS_DIR = settings.yolo_labels_dir


def make_yolo_annotations(coco_labels_path, images_dir, yolo_labels_dir):
    """
    Creating yolo annotation .txt files from coco csv file (labels_path).
    Each row in coco csv file has the structure image_filename,x_lt, y_lt, width, height,
    classname, object_id

    :param coco_labels_path: Path to a labels.csv file of coco annotations
    :param images_dir: Path to an images directory. Needed to get image shape and transform coco
    to yolo
    :param yolo_labels_dir: Path to directory, where yolo .txt files will be stored.
    So create labels directory first
    :return: None
    """

    coco_anns = make_coco_dict(coco_labels_path, images_dir)
    count = 0

    for image_name, annotations in coco_anns.items():
        yolo_annotations = []
        count += 1

        if count % 5000 == 0:
            print("5000...")

        for coco_annotation in annotations:
            yolo_class, *coco_bbox = coco_annotation

            img_path = os.path.join(images_dir, image_name)
            img_width, img_height = cv2.imread(img_path).shape[:2][::-1]
            yolo_bbox = coco_to_yolo(*coco_bbox, img_width, img_height)

            yolo_bbox = [str(el) for el in yolo_bbox]
            yolo_annotation = str(yolo_class) + " " + " ".join(yolo_bbox)
            yolo_annotations.append(yolo_annotation)

        yolo_annotations_text = "\n".join(yolo_annotations)
        label_name = image_name.replace("jpg", "txt")
        label_path = os.path.join(yolo_labels_dir, label_name)

        with open(label_path, "w") as label_file:
            label_file.write(yolo_annotations_text)


def make_coco_dict(coco_labels_path, images_dir):
    """
    Creating dictionary with coco annotations.

    :param coco_labels_path: Path to a labels.csv file of coco annotations
    :param images_dir: Path to an images directory. Needed to get image shape and transform coco
    to yolo
    :return: Dictionary with structure {image_filename: [[object_class, x_start, y_start, width, height]]}
    """

    images_anns = {image: [] for image in os.listdir(images_dir)}
    yolo_classes = {}
    yolo_classes_counter = 0

    with open(coco_labels_path) as f:
        _ = f.readline()

        for annotation in f:
            filename, x_from, y_from, width, height, sign_class, _ = annotation.split(",")

            if sign_class not in yolo_classes:
                yolo_classes[sign_class] = yolo_classes_counter
                yolo_classes_counter += 1

            images_anns[filename].append([
                yolo_classes[sign_class], int(x_from), int(y_from), int(width), int(height)
            ])
    return images_anns


def coco_to_yolo(x_start, y_start, width, height, img_width, img_height):
    """
    Making yolo bounding box from coco bounding box

    :param x_start: Top-left x
    :param y_start: Top-left y
    :param width: Bounding box width
    :param height: Bounding box height
    :param img_width: Image width
    :param img_height: Image height
    :return: Yolo bounding bbox. [x-c, y-c, w, h] Center coordinates & width & height
    """
    return pbx.convert_bbox((x_start, y_start, width, height),
                            from_type="coco", to_type="yolo", image_size=(img_width, img_height))


if __name__ == "__main__":
    make_yolo_annotations(COCO_LABELS_PATH, IMAGES_DIR, YOLO_LABELS_DIR)
