import glob

import cv2
import numpy as np

from Classes.BoundingBox import BoundingBox
from Classes.Frame import Frame
from PATHS import AICITY_ROOT, YOLO_ROOT

# Paths and constants setup
path = AICITY_ROOT + "vdo.avi"
path_frames = AICITY_ROOT + "frames/*.jpg"
yolo_classes = YOLO_ROOT + 'yolov3.txt'
yolo_weights = YOLO_ROOT + 'yolov3.weights'
yolo_config = YOLO_ROOT + 'yolov3.cfg'
imgs_path = './M6/detections/'
BEGIN_FRAME = 230


# Yolo functions
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_bbox(img, id, x, y, x_plus_w, y_plus_h):
    label = str(id)
    color = [255, 0, 0]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)


classes = None
with open(yolo_classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

classes = classes[:3]
COLORS = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
net = cv2.dnn.readNet(yolo_weights, yolo_config)
conf_threshold = 0.5
nms_threshold = 0.4

frames = []

framepath = glob.glob(path_frames)
for frame in framepath[BEGIN_FRAME:]:
    cFrame = Frame()
    image = cv2.imread(frame)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []

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
    for key, i in enumerate(indices):
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        bbox = BoundingBox(x, y, w, h)
        # cFrame.addDetection(key, bbox)
        cFrame.addDetection(key, bbox)

    frames.append(cFrame)
    idx = frames.index(cFrame)
    if idx != 0:
        frames[idx].track(frames[idx - 1])

    for idx, detection in cFrame.det.items():
        x = detection.top
        y = detection.left
        width = detection.width
        height = detection.height
        id = idx
        draw_bbox(image, id, int(x), int(y), int(x + width), int(y + height))

    # TODO: Tracking here, add to CFrame class detections the box ;)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("object detection", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
