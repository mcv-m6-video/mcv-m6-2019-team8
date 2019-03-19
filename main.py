"""
Testing the Detection Metrics, Week 1 Task
"""

import os
import cv2
import numpy as np
import collections
import matplotlib.pyplot as plt

from Utilities.functions import bb_intersect_over_union, f1_score, noiseboxlist, addnotation_to_bboxlist, read_images, \
    frames_to_image, remove_image

from Classes.BoundingBox import BoundingBox
from Classes.Frame import Frame

# from M6.yolo_opencv import get_output_layers, draw_prediction

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
# ADNOTATED_START_FRAME = 218
# ADNOTATED_END_FRAME = 247
# adn = addnotation_to_bboxlist("/home/marcinmalak/Desktop/m6repo/Adnotations",
#                               ADNOTATED_START_FRAME,
#                               ADNOTATED_END_FRAME)
# for i in range(0, ADNOTATED_END_FRAME - ADNOTATED_START_FRAME):
#     gt[ADNOTATED_START_FRAME + i] = adn[i]

# Setting font
font = cv2.FONT_HERSHEY_SIMPLEX

# Skipping to frame 218, because ground truth starts there:
cap.set(cv2.CAP_PROP_POS_FRAMES, 218)

# # Init arrays for plots
ioulist = []
framelist = []
ioulistAddn = []
framelistAddn = []

yolo_classes = '/home/marcinmalak/Desktop/m6-group8/mcv-m6-2019-team8/M6/yolov3.txt'
yolo_weights = '/home/marcinmalak/Desktop/m6-group8/mcv-m6-2019-team8/M6/yolov3.weights'
yolo_config = '/home/marcinmalak/Desktop/m6-group8/mcv-m6-2019-team8/M6/yolov3.cfg'
imgs_path = './M6/detections/'


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)


while cap.isOpened():
    ret, frame = cap.read()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1280, 720)
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frameText = "Frame:" + str(currentFrame)
    cv2.putText(frame, frameText, (0, 50), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    name = './M6/detections/' + str(int(currentFrame)) + '.png'
    print('Creating...' + name)
    cv2.imwrite(name, frame)
    image = cv2.imread(name)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(yolo_classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(yolo_weights, yolo_config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("object detection", image)

    # TODO: LOOKS LIKE A SHITTY MOLOH, but it works. We should consider change the code in the readable way!
    #  It's possible to generate video separately. Remove image function remove every img created during
    #  the prediction, after prediction. It's possible to generate both groundtrouth with prediction on the same screen
    #  just #  (delete 'cv2.namedWindow('image', cv2.WINDOW_NORMAL)'

    # -------------------------------------------------------------------------------------------------

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
    # if currentFrame > 218 and currentFrame < 247:
    #     if gt.get(currentFrame, None) and det.get(currentFrame, None):
    #         cFrame = Frame(currentFrame, gt[currentFrame], det[currentFrame])
    #         groundtruth = cFrame.groundtruth()
    #         if isinstance(groundtruth, list):
    #             for bbox in groundtruth:
    #                 cv2.rectangle(frame, bbox.topleft(), bbox.bottomright(), (0, 255, 0), 3)
    #                 result = cFrame.to_result(0.5)
    #         elif isinstance(groundtruth, BoundingBox):
    #             cv2.rectangle(frame, groundtruth.topleft(), groundtruth.bottomright(), (0, 255, 0), 3)
    #         iou = cFrame.meaniou()
    #         ioulistAddn.append(iou)
    #         framelistAddn.append(currentFrame)
    #         # print(str(currentFrame)+":"+str(iou))

    cv2.imshow('image', frame)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    remove_image(imgs_path, currentFrame)

    # Plot for IoU over frame
    # fig = plt.figure("IoU Plot")
    # plt.title('IoU(frame)')
    # plt.xlabel('Video frame')
    # plt.ylabel('IoU Value')
    # plt.plot(framelistAddn, ioulistAddn)
    # plt.show()

    # # Plot for F1 score per frame
    # fig = plt.figure("F1 Plot")
    # plt.title('F1(frame)')
    # plt.xlabel('Video frame')
    # plt.ylabel('F1 Measure')
    # plt.plot(frameVals, F1Vals)
    # plt.show()

    # plt.show()
cap.release()
cv2.destroyAllWindows()
