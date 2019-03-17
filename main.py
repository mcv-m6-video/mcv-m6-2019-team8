"""
Testing the Detection Metrics, Week 1 Task
"""

import os
import cv2
import collections
import matplotlib.pyplot as plt
from Utilities.functions import bb_intersect_over_union, f1_score, noiseboxlist, addnotation_to_bboxlist
from Classes.BoundingBox import BoundingBox
from Classes.Frame import Frame

# Seriously, we should add some relative paths or path file
if os.name == 'nt':
    path = 'E:/DATA_Projects/M6/Week1/Data/AICity_data/AICity_data/train/S03/c010/vdo.avi'
    path_gt = 'E:/DATA_Projects/M6/Week1/Data/AICity_data/AICity_data/train/S03/c010/gt/gt.txt'
    path_det = 'E:/DATA_Projects/M6/Week1/Data/AICity_data/AICity_data/train/S03/c010/det/det_ssd512.txt'
else:
    path = '/home/marcinmalak/Desktop/M6 Video Analysis/AICity_data/AICity_data/train/S03/c010/vdo.avi'
    path_gt = '/home/marcinmalak/Desktop/M6 Video Analysis/AICity_data/AICity_data/train/S03/c010/gt/gt.txt'
    path_det = '/home/marcinmalak/Desktop/M6 Video Analysis/AICity_data/AICity_data/train/S03/c010/det/det_yolo3.txt'

# Importing files
cap = cv2.VideoCapture(path)

# Importing Ground truth
gt = {}
with open(path_gt) as f:
    for line in f:
        data = line.split(",")
        gt[int(data[0])] = [BoundingBox(int(data[2]),
                                        int(data[3]),
                                        int(data[4]),
                                        int(data[5]))]
# Import Detection
det = collections.defaultdict(list)
with open(path_det) as f:
    for line in f:
        data = line.split(",")
        row = BoundingBox(int(float(data[2])),
                          int(float(data[3])),
                          int(float(data[4])),
                          int(float(data[5])))
        det[int(data[0])].append(row)

# Import ground-truth addnotations
ADNOTATED_START_FRAME = 218
ADNOTATED_END_FRAME = 247
adn = addnotation_to_bboxlist("/home/marcinmalak/Desktop/m6repo/Adnotations",
                              ADNOTATED_START_FRAME,
                              ADNOTATED_END_FRAME)
for i in range(0, ADNOTATED_END_FRAME - ADNOTATED_START_FRAME):
    gt[ADNOTATED_START_FRAME + i] = adn[i]

# Setting font
font = cv2.FONT_HERSHEY_SIMPLEX

# Skipping to frame 218, because ground truth starts there:
cap.set(cv2.CAP_PROP_POS_FRAMES, 218)

# # Init arrays for plots
ioulist = []
framelist = []
ioulistAddn = []
framelistAddn = []

while cap.isOpened():
    ret, frame = cap.read()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1280, 720)
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frameText = "Frame:" + str(currentFrame)
    cv2.putText(frame, frameText, (0, 50), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    # Task 1, IoU on Noised
    if gt.get(currentFrame, None) and det.get(currentFrame, None):
        cFrame = Frame(currentFrame, gt[currentFrame], det[currentFrame])
        groundtruth = cFrame.groundtruth()
        if isinstance(groundtruth, list):
            for bbox in groundtruth:
                cv2.rectangle(frame, bbox.topleft(), bbox.bottomright(), (0, 255, 0), 3)
                result = cFrame.to_result(0.5)
        elif isinstance(groundtruth, BoundingBox):
            cv2.rectangle(frame, groundtruth.topleft(), groundtruth.bottomright(), (0, 255, 0), 3)
        iou = cFrame.meaniou()
        rez = cFrame.to_result(0.5)
        ioulist.append(iou)
        framelist.append(currentFrame)
        print(str(currentFrame) + ":" + str(iou) + ":" + str(rez))

    # Task 2, Adnotated frames
    if currentFrame > 218 and currentFrame < 247:
        if gt.get(currentFrame, None) and det.get(currentFrame, None):
            cFrame = Frame(currentFrame, gt[currentFrame], det[currentFrame])
            groundtruth = cFrame.groundtruth()
            if isinstance(groundtruth, list):
                for bbox in groundtruth:
                    cv2.rectangle(frame, bbox.topleft(), bbox.bottomright(), (0, 255, 0), 3)
                    result = cFrame.to_result(0.5)
            elif isinstance(groundtruth, BoundingBox):
                cv2.rectangle(frame, groundtruth.topleft(), groundtruth.bottomright(), (0, 255, 0), 3)
            iou = cFrame.meaniou()
            ioulistAddn.append(iou)
            framelistAddn.append(currentFrame)
            # print(str(currentFrame)+":"+str(iou))

    cv2.imshow('image', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
#
# Plot for IoU over frame
fig = plt.figure("IoU Plot")
plt.title('IoU(frame)')
plt.xlabel('Video frame')
plt.ylabel('IoU Value')
plt.plot(framelistAddn, ioulistAddn)
plt.show()

# # Plot for F1 score per frame
# fig = plt.figure("F1 Plot")
# plt.title('F1(frame)')
# plt.xlabel('Video frame')
# plt.ylabel('F1 Measure')
# plt.plot(frameVals, F1Vals)
# plt.show()

plt.show()
cap.release()
cv2.destroyAllWindows()
