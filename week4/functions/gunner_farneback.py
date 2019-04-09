import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PATHS import RESULT_DIR


def gunner_farneback(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def evaluate_gunner_farneback(names):
    gt = cv2.imread(names[0], -1)
    prvs = cv2.imread(names[1], 0)
    curr = cv2.imread(names[2], 0)

    GUN_opt_flow = gunner_farneback(prvs, curr)
    heigh = GUN_opt_flow.shape[0]
    width = GUN_opt_flow.shape[1]
    best_opt_flow = np.concatenate((GUN_opt_flow, np.ones((heigh, width, 1))), axis=2)
    best_mmse, best_pepn = compute(gt, best_opt_flow, prvs.shape[0], prvs.shape[1])

    return best_mmse, best_pepn


def compute(gt, test, offset_y, offset_x):
    gtx = (np.array(gt[:, :, 1], dtype=float) - (2 ** 15)) / 64.0
    gty = (np.array(gt[:, :, 2], dtype=float) - (2 ** 15)) / 64.0
    gtz = np.array(gt[:, :, 0], dtype=bool)
    print("non ocluded gt:" + str(np.count_nonzero(gtz)))

    testx = (np.array(test[:, :, 0], dtype=float)) / offset_x
    testy = (np.array(test[:, :, 1], dtype=float)) / offset_y
    testz = np.array(test[:, :, 2], dtype=bool)
    print("non ocluded test:" + str(np.count_nonzero(testz)))

    mask1 = np.logical_and(gtz, testz)

    validPixels = np.count_nonzero(mask1)
    print("Valid pixels " + str(validPixels))
    if np.count_nonzero(mask1) < int(0.2 * np.prod(gt.shape)):
        warnings.warn("Low number of valid pixels")

    gtx_1 = gtx * mask1
    gty_1 = gty * mask1
    testx_1 = testx * mask1
    testy_1 = testy * mask1

    # Mean Square error in Non-ocluded areas
    msen = np.sqrt((gtx_1 - testx_1) ** 2 + (gty_1 - testy_1) ** 2)
    msen_r = np.reshape(msen, [-1])[np.reshape(mask1, [-1])]

    plt.figure(1)
    plt.hist(msen_r, bins=50)
    plt.title("MSE normalized histogram")
    plt.ylabel("% pixels")
    plt.xlabel("MSE")
    plt.show()

    m_msen = np.mean(np.mean(msen_r))
    print("MMSE (non-ocluded): " + str(m_msen))

    plt.figure(2)
    plt.imshow(msen)
    plt.colorbar()
    plt.title("MSE map")
    plt.show()

    mask2 = msen > 3.0

    # Percentage of Erroneous Pixels in Non-occluded areas
    pepn = np.count_nonzero(mask1[mask2]) / np.count_nonzero(mask1)
    print("PEPN (non-ocluded): " + str(pepn) + "\n")

    return m_msen, pepn
