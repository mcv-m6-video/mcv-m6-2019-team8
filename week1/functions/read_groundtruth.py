from typing import List

from week1.models.BoundingBox import BoundingBox


def read_groundtruth(path):
    # [frame, -1, left, top, width, height, conf, -1, -1, -1]
    gt = {}
    with open(path) as f:
        for line in f:
            data = line.split(",")
            gt[int(data[0])] = [BoundingBox(int(data[2]),
                                            int(data[3]),
                                            int(data[4]),
                                            int(data[5]))]
    return gt
