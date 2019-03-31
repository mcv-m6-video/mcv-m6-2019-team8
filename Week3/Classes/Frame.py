from Classes.BoundingBox import BoundingBox
from typing import List
from Utilities.functions import bb_intersect_over_union
import numpy as np

"""
    Inspired by Team 5 frame model, thanks!
    https://github.com/mcv-m6-video/mcv-m6-2019-team5/blob/master/src/model/frame.py
"""


class Frame:
    # def __init__(self, number: int, gt: List[BoundingBox], det: List[BoundingBox]):
    def __init__(self):
        self.gt = {}
        self.det = {}

    def addDetections(self, det: List[BoundingBox]):
        """
        :param det: detection bounding boxes list in the frame
        :return: finish code
        """
        self.det = det
        return True

    def addDetection(self,key, bbox: BoundingBox):
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

    def iou(self) -> List[float]:
        ret = []
        for gtbox in self.gt:
            max_iou = 0
            for detbox in self.det:
                # iou = detbox.iou(gtbox)
                iou = bb_intersect_over_union(gtbox, detbox)
                if iou > max_iou:
                    max_iou = iou
            ret.append(max_iou)
        return ret

    def meaniou(self):
        """
        :return: returns mean iou from frame
        """
        ret = self.iou()
        if len(ret) > 0:
            return float(np.mean(ret))
        else:
            return 0

    def to_result(self, threshold: float) -> ():
        """
        :param threshold: acceptable iou value for TP
        :return: returns touple with tp, fp, fn values
        """
        tp = 0
        for gtbox in self.gt:
            for detbox in self.det:
                if detbox.iou(gtbox) > threshold:
                    tp += 1
                    break

        fp = len(self.det) - tp
        fn = len(self.gt) - tp
        ret = (tp, fp, 0, fn)
        return ret

    def track(self, other: 'Frame'):
        ret = {}
        old = other.det
        max_key_old = max(other.det.keys())
        for detection in self.det.values():
            max_iou = 0
            max_key = None
            for key, olddetection in old.items():
                iou = detection.iou(olddetection)
                if iou > max_iou:
                    max_iou = iou
                    max_key = key
            ret[max_key] = detection
            # del old[max_key]


        # for detection in enumerate(self.det):
        #     max_iou = 0
        #     max_overlap_index = None
        #     for idx, olddetection in enumerate(other.det):
        #         iou = detection.iou(olddetection)
        #         if iou > max_iou:
        #             max_iou = iou
        #             max_overlap_index = idx
        #     ret[max]






        self.det = ret
        return True


