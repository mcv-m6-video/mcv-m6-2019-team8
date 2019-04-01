import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importing frames and display it
from PATHS import PATH_FRAMES, PATH_GT, PATH_DETECTION, PATH_VIDEO, PATH_OPTICAL_FLOW, PATH_OPTICAL_FLOW_GT, \
    RESULT_DIR, PATH_OF_IMAGES
from week1.functions.OpticalFlowCalculations import oP_calculations, readOF, readOFimages, OFplots
from week1.functions.read_detections import read_detection
from week1.functions.read_groundtruth import read_groundtruth
from week1.models.BoundingBox import BoundingBox
from week1.models.Frame import Frame

detections = {'1': 'det_mask_rcnn.txt',
              '2': 'det_ssd512.txt',
              '3': 'det_yolo3.txt'}

path_video = PATH_VIDEO
path_frames = PATH_FRAMES
path_gt = PATH_GT
path_dt = PATH_DETECTION + detections['1']
BEGIN_FRAME = 217

framespath = glob.glob(path_frames)

# Create dict for Detections and display it
det = read_detection(path_dt)
# Create dict for GroundTruth and display it
gt = read_groundtruth(path_gt)

cap = cv2.VideoCapture(path_video)
cap.set(cv2.CAP_PROP_POS_FRAMES, BEGIN_FRAME)

frames = []

# TASK 1 #

# Displaying groundtruth with detection metrics

while cap.isOpened():
    ret, frame = cap.read()
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frameText = "Frame:" + str(currentFrame)
    cv2.putText(frame, frameText, (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (200, 255, 0), 2, cv2.LINE_AA)

    if gt.get(currentFrame, None) and det.get(currentFrame, None):
        cFrame = Frame(currentFrame, gt[currentFrame], det[currentFrame])
        groundtruth = cFrame.groundtruth()
        if isinstance(groundtruth, list):
            for bbox in groundtruth:
                cv2.rectangle(frame, bbox.topleft(), bbox.bottomright(), (255, 0, 0), thickness=5)
        elif isinstance(groundtruth, BoundingBox):
            cv2.rectangle(frame, groundtruth.topleft(), groundtruth.bottomright(), (0, 255, 0), thickness=5)
    img_resize = cv2.resize(frame, (960, 600))
    cv2.imshow('Video frames', img_resize)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Task3 #

# Optical flow evaluation metrics

flowResultFrame45 = cv2.imread(PATH_OPTICAL_FLOW + 'LKflow_000045_10.png', -1)
flowResultFrame157 = cv2.imread(PATH_OPTICAL_FLOW + 'LKflow_000157_10.png', -1)
flowGroundTruthFrame45 = cv2.imread(PATH_OPTICAL_FLOW_GT + '000045_10.png', -1)
flowGroundTruthFrame157 = cv2.imread(PATH_OPTICAL_FLOW_GT + '000157_10.png', -1)

dist45 = (np.asarray(oP_calculations(flowResultFrame45, flowGroundTruthFrame45, 45)[2])).reshape(
    (flowResultFrame45.shape[0], flowResultFrame45.shape[1]))
dist157 = (np.asarray(oP_calculations(flowResultFrame157, flowGroundTruthFrame157, 157)[2])).reshape(
    (flowResultFrame157.shape[0], flowResultFrame157.shape[1]))

# Plotting #
dist45 = np.ma.masked_where(dist45 == 0, dist45)
cmap = plt.cm.plasma
cmap.set_bad(color='red')

plt.imshow(dist45, interpolation='none', cmap=cmap)
plt.colorbar(orientation='horizontal')
plt.savefig(RESULT_DIR + 'plot45.png')
plt.show()

dist157 = np.ma.masked_where(dist157 == 0, dist157)
cmap = plt.cm.plasma
cmap.set_bad(color='red')

plt.imshow(dist157, interpolation='none', cmap=cmap)
plt.colorbar(orientation='horizontal')
plt.savefig(RESULT_DIR + 'plot157.png')
plt.show()
plt.close()

# Task5 #

# Visual representation optical flow

ofImages = readOF(PATH_OPTICAL_FLOW)
images = readOFimages(PATH_OF_IMAGES)

OFplots(ofImages, images)
