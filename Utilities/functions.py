"""
This module holds all the utility functions.
"""
import glob
import math
from PIL import Image

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from Classes.BoundingBox import BoundingBox
from typing import List


def bb_intersect_over_union(boxA: BoundingBox, boxB: BoundingBox):
    """
    Calculates and returns IoU value of two
    provided parameters that are
    rectangular regions.
    """
    boxA = [boxA.top, boxA.left, boxA.top + boxA.width, boxA.left + boxA.height]
    boxB = [boxB.top, boxB.left, boxB.top + boxB.width, boxB.left + boxB.height]

    # Getting X,Y of Intersection Rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    # Compute area of the rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute area of pred box and gt box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute IoU
    # Intersection area / Sum (gt+pred) area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # idk ;_;
    iou = 1 - math.modf(iou)[0]

    return iou


def f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def noiseboxlist(bboxlist: List[BoundingBox], c_app: float, c_change: float, magnitude: int):
    """
    Function generates some noise for bounding box list
    :param bboxlist: bounding box list
    :param c_app: chance of appearing random box
    :param c_change: chance of change happening
    :param scale: magnitutde of change, multiplication
    :return: returns the modified list
    """
    bboxnew = []
    for bbox in bboxlist:
        if np.random.random_sample() < c_change:
            bboxnew.append(bbox)
            pass
        nbox = BoundingBox(0, 0, 0, 0)
        nbox.top = int(bbox.top + np.random.random_sample() * magnitude)
        nbox.left = int(bbox.left + np.random.random_sample() * magnitude)
        nbox.width = int(bbox.width * np.random.random_sample() * magnitude)
        nbox.height = int(bbox.height * np.random.random_sample() * magnitude)
        bboxnew.append(nbox)

        if np.random.random_sample() < c_app:
            bboxnew.append(BoundingBox(np.random.random_sample(),
                                       np.random.random_sample(),
                                       np.random.random_sample(),
                                       np.random.random_sample()))
    return bboxnew


def addnotation_to_bboxlist(root_directory: str, start: int, end: int) -> List[List[BoundingBox]]:
    """
    Special shout-out to Team 5 again, we function copied from their repo
    https://github.com/mcv-m6-video/mcv-m6-2019-team5/blob/master/src/utils/read_annotations.py
    :param root_directory: directory to addnotation folder
    :param start: starting index of addnotation
    :param end: finishing index of addnotation
    :return: returns the list (for each frame) of detetcions (also a list)
    """
    frames_detections: List[List[BoundingBox]] = []
    for i in range(start, end + 1):
        frame_path = 'vdo{:04d}.xml'.format(i)
        root = ET.parse(os.path.join(root_directory, frame_path)).getroot()
        frame_detections: List[BoundingBox] = []
        for obj in root.findall('object'):
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            frame_detections.append(BoundingBox(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
        frames_detections.append(frame_detections)
    return frames_detections


def frames_to_image(frame, currentFrame):
    name = './M6/detections/' + str(int(currentFrame)) + '.png'
    print('Creating...' + name)
    cv2.imwrite(name, frame)


def read_images(path) -> list:
    image_list = []
    for filename in glob.glob(path + '*.png'):
        image_list.append(filename)

    return image_list


def remove_image(path, img_name):
    os.remove(path + str(int(img_name)) + '.png')
    # check if file exists or not
    if os.path.exists(path + str(int(img_name)) + '.png') is False:
        # file did not exists
        return True
