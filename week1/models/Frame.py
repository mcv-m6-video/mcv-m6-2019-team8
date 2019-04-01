from typing import List

import numpy as np

from week1.models.BoundingBox import BoundingBox

"""
    Inspired by Team 5 frame model, thanks!
    https://github.com/mcv-m6-video/mcv-m6-2019-team5/blob/master/src/model/frame.py
"""


class Frame:
    def __init__(self, number: int, gt: List[BoundingBox], det: List[BoundingBox]):
        """
        :param number: frame number
        :param gt: ground truth bounding boxes list in the frame
        :param det: detection bounding boxes list in the frame
        """
        self.number = number
        self.gt = gt
        self.det = det

    def addDetections(self, det: List[BoundingBox]):
        """
        :param det: detection bounding boxes list in the frame
        :return: finish code
        """
        self.det = det
        return True

    def addDetection(self, key, bbox: BoundingBox):
        """
        :param bbox: bounding box to be added to the list
        :return: finish code
        """
        self.det[key] = bbox
        return True

    def addGroundtruth(self, gt: List[BoundingBox]):
        """
        :param gt: ground truth bounding boxes list in the frame
        :return: finish code
        """
        self.gt = gt
        return True

    def groundtruth(self):
        '''
        :return: ground truth list for frame
        '''
        return self.gt

    def detections(self):
        '''
        :return: detection list for frame
        '''
        return self.det

    # def iou(self) -> List[float]:
    #     ret = []
    #     for gtbox in self.gt:
    #         max_iou = 0
    #         for detbox in self.det:
    #             # iou = detbox.iou(gtbox)
    #             iou = bb_intersect_over_union(gtbox, detbox)
    #             if iou > max_iou:
    #                 max_iou = iou
    #         ret.append(max_iou)
    #     return ret
