import collections

from week1.models.BoundingBox import BoundingBox


def read_detection(path):
    det = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            data = line.split(",")
            row = BoundingBox(int(float(data[2])),
                              int(float(data[3])),
                              int(float(data[4])),
                              int(float(data[5])))
            det[int(data[0])].append(row)

    return det
