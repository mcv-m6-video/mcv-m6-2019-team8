import numpy as np
from PIL import Image
import time
import argparse
from pyflow import pyflow
import cv2
from week4.functions.OpticalFlowCalculations import readOF, readOFimages, block_matching, flow_error, \
    flow_visualization, flow_read

def coarse2Fine(prev_frame, curr_frame):
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_frame_gray = prev_frame_gray[:, :, np.newaxis]
    curr_frame_gray = curr_frame_gray[:, :, np.newaxis]

    prev_array = np.asarray(prev_frame_gray)
    curr_array = np.asarray(curr_frame_gray)

    # prev_array = np.asarray(prev_frame)
    # curr_array = np.asarray(curr_frame)

    prev_array = prev_array.astype(float) / 255.
    curr_array = curr_array.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        prev_array, curr_array, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, prev_array.shape[0], prev_array.shape[1], prev_array.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    np.save('../result/pyflow/outFlow.npy', flow)

    hsv = np.zeros(prev_frame.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('../result/pyflow/outFlow_new.png', rgb)
    cv2.imwrite('../result/pyflow/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)
    cv2.imshow("optical flow", rgb)
    cv2.waitKey()
    return flow



if __name__ == '__main__':

    F_gt = flow_read("../data/kitti/gt/000045_10.png")
    frame1_rgb = cv2.imread('../data/kitti/000045_10.png')
    frame2_rgb = cv2.imread('../data/kitti/000045_11.png')
    flow = coarse2Fine(frame1_rgb, frame2_rgb)
    flow_error(F_gt, flow)


