import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

from PATHS import RESULT_DIR


def oP_calculations(opResult, opGT, frameName=0, plot=True, normInp=True):
    # To compute MSEN (Mean Square Error in Non-occluded areas)
    distances = []
    # Percentage of Erroneous Pixels in Non-occluded areas (PEPN)
    errPixels = []
    # Euclidean distance threshold for PEPN
    errorTh = 3
    errorImage = []

    # Loop through each pixel
    for i in range(np.shape(opResult)[0]):
        for j in range(np.shape(opResult)[1]):
            # Convert u-/v-flow into floating point values
            if normInp:
                u_float = (float(opResult[i][j][1]) - 2 ** 15) / 64.0
                v_float = (float(opResult[i][j][2]) - 2 ** 15) / 64.0
            else:
                u_float = float(opResult[i][j][1])
                v_float = float(opResult[i][j][2])
            u_gt_float = (float(opGT[i][j][1]) - 2 ** 15) / 64.0
            v_gt_float = (float(opGT[i][j][2]) - 2 ** 15) / 64.0
            # If ground truth is available, compare it to the estimation result using Euclidean distance
            if opGT[i][j][0] == 1:
                dist = np.sqrt((u_float - u_gt_float) ** 2 + (v_float - v_gt_float) ** 2)
                distances.append(dist)
                errorImage.append(dist)
                # If the distance is more than the threshold, consider the pixel as erroneous
                if abs(dist) > errorTh:
                    errPixels.append(True)
                else:
                    errPixels.append(False)
            else:
                errorImage.append(0)

    msen = np.mean(distances)
    pepn = np.mean(errPixels) * 100

    print("Mean Square Error in Non-occluded areas (MSEN): ", msen)

    print("Percentage of Erroneous Pixels in Non-occluded areas (PEPN): ", pepn, "%")

    # Print[, plot] and return results
    if frameName == 45:
        cm = plt.cm.plasma
        # Get the histogram
        Y, X = np.histogram(distances, 20)
        x_span = X.max() - X.min()
        C = [cm(((x - X.min()) / x_span)) for x in X]
        plt.bar(X[:-1], Y * 100, color=C, width=X[1] - X[0])
        plt.xlabel('Distance error')
        plt.ylabel('N. of pixels (%)')
        plt.title('SEG 45 - Histogram of Error per pixel')
        plt.savefig(RESULT_DIR + 'plot45.png')
        plt.show()
    elif frameName == 157:
        cm = plt.cm.plasma
        # Get the histogram
        Y, X = np.histogram(distances, 20)
        x_span = X.max() - X.min()
        C = [cm(((x - X.min()) / x_span)) for x in X]
        plt.bar(X[:-1], Y * 100, color=C, width=X[1] - X[0])
        plt.xlabel('Distance error')
        plt.ylabel('N. of pixels (%)')
        plt.title('SEG 157 - Histogram of Error per pixel')
        plt.savefig(RESULT_DIR + 'plot45.png')
        plt.show()

    return msen, pepn, errorImage


def readOF(ofPath):
    imgNames = os.listdir(ofPath)
    imgNames.sort()
    images = []
    for name in imgNames:
        if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
            images.append(cv2.imread(ofPath + name, -1))
    return images


def readOFimages(ofOrPath):
    imgNames = os.listdir(ofOrPath)
    imgNames.sort()
    ofImages = []
    for name in imgNames:
        if name.endswith('.png'):
            if int(name[7:9]) == 10:
                im = cv2.imread(ofOrPath + name)
                ofImages.append(im)
    return ofImages


def OFplots(ofImages, images):
    step = 10
    ind = 0

    for ofIm in ofImages:
        ofIm = cv2.resize(ofIm, (0, 0), fx=1. / step, fy=1. / step)
        rows, cols, depth = ofIm.shape
        U = []
        V = []

        for pixel in range(0, ofIm[:, :, 0].size):
            isOF = ofIm[:, :, 0].flat[pixel]
            if isOF == 1:
                U.append(((float(ofIm[:, :, 1].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
                V.append(((float(ofIm[:, :, 2].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
            else:
                U.append(0)
                V.append(0)

        U = np.reshape(U, (rows, cols))
        V = np.reshape(V, (rows, cols))
        x, y = np.meshgrid(np.arange(0, cols * step, step), np.arange(0, rows * step, step))

        cm = plt.cm.plasma
        plt.imshow(images[ind])
        plt.quiver(x, y, U, V, scale=0.1, alpha=1, color='r')
        plt.title('Optical Flow')
        plt.savefig(RESULT_DIR + 'OF' + str(ind) + '.png')
        plt.show()
        plt.close()
        ind += 1
