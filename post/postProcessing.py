from __future__ import division

import argparse
import json
import os

import cv2
import numpy as np

from scipy import spatial
from scipy import stats

from colorMapper import ColorMapper


def sub_threshold(img, st, erode_flag=False, unsharp_flag=False):
    if unsharp_flag:
        img = unsharp(img, st)
    ret, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if erode_flag:
        thresh = erode(thresh, st)
    return thresh


def threshold(img):
    # --- threshold on blue channel
    m1 = sub_threshold(img[:, :, 0], 1, True, True)
    # --- threshold on green channel
    m2 = sub_threshold(img[:, :, 1], 2, True, True)
    # --- threshold on red channel
    m3 = sub_threshold(img[:, :, 2], 3, True, True)

    # --- adding up all the results above ---
    res = cv2.add(m1, cv2.add(m2, m3))
    return res


def erode(thresh, st):
    kernel = np.ones((3, 4), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    return thresh


def unsharp(imgray, st):
    # Unsharp mask here
    imgray = imgray.copy()
    gaussian = cv2.GaussianBlur(imgray, (7, 7), 10.0)
    unsharp_image = cv2.addWeighted(imgray, 2.5, gaussian, -1.5, 0, imgray)
    return unsharp_image


def get_bounding_boxes(image_path, dst_img_path, dst_json_path):
    elements = []
    image = cv2.imread(image_path)
    original = image.copy()
    thresh = threshold(image)
    new_semantic = np.ones_like(original) * 255

    # Find contours, obtain bounding box, extract and save ROI
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI = original[y:y + h, x:x + w]
        dominant_color, label = get_nearest_dominant_color(ROI)

        if dominant_color is None:
            continue

        cv2.rectangle(new_semantic, (x, y), (x + w, y + h), dominant_color, 3)
        elements.append({"points": [[x, y], [x + w, y + h]], "label": label})

    # Generate image filename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Save images
    cv2.imwrite(os.path.join(dst_img_path, base_filename + "0.png"), image)
    cv2.imwrite(os.path.join(
        dst_img_path, base_filename + "1.png"), new_semantic)

    # Save JSON
    json_path = os.path.join(dst_json_path, base_filename + ".json")
    create_json_file(json_path, elements)


def get_nearest_dominant_color(img):
    pixels = img.reshape(-1, 3)

    # Check if there are enough pixels
    if len(pixels) < 50:
        return None, None

    try:
        # Use scipy KDTree to find nearest colors
        _, ind = kdt.query(pixels)

        # Use numpy to find the most frequent index
        unique, counts = np.unique(ind, return_counts=True)
        most_frequent_index = unique[np.argmax(counts)]

        # Get the closest color and label
        closest_color = color_np_list[most_frequent_index]
        label = labels[most_frequent_index]

        return (
            int(closest_color[0]),
            int(closest_color[1]),
            int(closest_color[2])
        ), label

    except Exception as e:
        print(f"Error processing color: {e}")
        return None, None


def create_json_file(path, elements):
    data = {"shapes": elements,
            "imageHeight": 567,
            "imageWidth": 360
            }
    if data is not None and len(data) > 0:
        with open(path, "w+") as ff:
            json.dump(data, ff, indent=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_images_folder", "-s", help="path to generated semantic images",
                        default="./samples")
    parser.add_argument("--destination_path", "-d", help="path to destination folder",
                        default="./results")
    parser.add_argument("--color_map_file", "-c", help="path to file for mapping of color to UI element type",
                        default="./resources/ui_labels_color_map.csv")

    args = parser.parse_args()

    folder_path = args.semantic_images_folder
    dest_path = args.destination_path
    color_map_file = args.color_map_file

    # Create destination subdirectories
    dst_img_path = os.path.join(dest_path, "img")
    dst_json_path = os.path.join(dest_path, "json")

    # Create destination directories if they don't exist
    os.makedirs(dst_img_path, exist_ok=True)
    os.makedirs(dst_json_path, exist_ok=True)

    # Read color map
    color_map = ColorMapper.read_label_color_map(color_map_file, bgr=False)
    color_np_list = np.array(list(color_map.values()))
    labels = list(color_map.keys())
    kdt = spatial.KDTree(color_np_list)

    # Process all images in the input folder
    for file in os.listdir(folder_path):
        # Check if it's an image file
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, file)
            get_bounding_boxes(image_path, dst_img_path, dst_json_path)
