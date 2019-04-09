import time

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import ndimage

from PATHS import RESULT_DIR


def readOFimages(ofOrPath):
    imgNames = os.listdir(ofOrPath)
    imgNames.sort()
    ofImages = []
    for name in imgNames:
        if name.endswith('.png'):
            ofImages.append(cv2.cvtColor(cv2.imread(ofOrPath + name), cv2.COLOR_BGR2GRAY))
    return ofImages


def readOF(ofPath):
    imgNames = os.listdir(ofPath)
    imgNames.sort()
    images = []
    for name in imgNames:
        if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
            images.append(cv2.imread(ofPath + name, -1))
    return images


def mse(array1, array2):
    return np.mean(np.power(array1.astype(np.float32) - array2.astype(np.float32), 2))


def sse(array1, array2):
    return np.sum(np.power(array1.astype(np.float32) - array2.astype(np.float32), 2))


"""Based on the previous year challenge"""


def block_matching(im1, im2, block_size=(3, 3), step=(1, 1), area=(2 * 3 + 3, 2 * 3 + 3), area_step=(1, 1),
                   error_func=mse, error_thresh=1, verbose=False):
    if im1.shape != im2.shape:
        print('ERROR: Image shapes are not the same!')
        exit(-1)

    rows, cols = im1.shape[:2]

    odd_block = (block_size[0] % 2, block_size[1] % 2)
    halfs_block = (block_size[0] / 2, block_size[1] / 2)
    padding = (halfs_block[0], halfs_block[1])

    odd_area = (area[0] % 2, area[1] % 2)
    halfs_area = (area[0] / 2, area[1] / 2)

    im1 = cv2.copyMakeBorder(im1, int(padding[0]), int(padding[0]), int(padding[1]), int(padding[1]),
                             cv2.BORDER_REFLECT)
    im2 = cv2.copyMakeBorder(im2, int(padding[0]), int(padding[0]), int(padding[1]), int(padding[1]),
                             cv2.BORDER_REFLECT)

    if len(im1.shape) == 2:
        im1 = np.copy(im1)[:, :, np.newaxis]
        im2 = np.copy(im2)[:, :, np.newaxis]

    result = np.empty([rows, cols, 2])  # step

    # IM1's double loop
    result_i = 0
    for i in np.arange(padding[0], rows + padding[0], step=step[0]):
        start = time.time()
        result_j = 0
        for j in np.arange(padding[1], cols + padding[1], step=step[1]):
            block1 = im1[int(i) - int(halfs_block[0]):int(i) + int(halfs_block[0]) + int(odd_block[0]),
                     int(j) - int(halfs_block[1]):int(j) + int(halfs_block[1]) + int(odd_block[1]), :]

            area_range = ((i - halfs_area[0] if i - halfs_area[0] > padding[0] else padding[0],
                           i + halfs_area[0] + odd_area[0] if i + halfs_area[0] + odd_area[0] < rows + padding[0] \
                               else rows + padding[0]),
                          (j - halfs_area[1] if j - halfs_area[1] > padding[1] else padding[1],
                           j + halfs_area[1] + odd_area[1] if j + halfs_area[1] + odd_area[1] < cols + padding[1] \
                               else cols + padding[1]))

            block2 = im2[int(i) - int(halfs_block[0]):int(i) + int(halfs_block[0]) + int(odd_block[0]),
                     int(j) - int(halfs_block[1]):int(j) + int(halfs_block[1]) + int(odd_block[1]), :]

            no_flow_error = error_func(block1, block2)
            min_error = no_flow_error
            max_error = min_error

            # IM2's double loop
            k_vector = np.arange(area_range[0][0],
                                 area_range[0][1], step=area_step[
                    0])
            l_vector = np.arange(area_range[1][0],
                                 area_range[1][1], step=area_step[
                    1])
            for k in k_vector:
                for l in l_vector:
                    if k == i and j == l:
                        continue

                    block2 = im1[int(k) - int(halfs_block[0]):int(k) + int(halfs_block[0]) + int(odd_block[0]),
                             int(l) - int(halfs_block[1]):int(l) + int(halfs_block[1]) + int(odd_block[1]), :]
                    # Invalid dimension for image

                    cur_error = error_func(block1, block2)

                    if cur_error < min_error:
                        min_error = cur_error
                        move = (k - i, l - j)
                    if cur_error > max_error:
                        max_error = cur_error
            if np.abs(min_error - no_flow_error) < error_thresh:
                move = (0, 0)
            result[result_i:result_i + step[0], result_j:result_j + step[1], 0] = move[0]
            result[result_i:result_i + step[0], result_j:result_j + step[1], 1] = move[1]

            result_j += step[1]
        if verbose:
            print('Processed image row: {}/{} spent time: {}sec'.format(result_i, rows, time.time() - start))
        result_i += step[0]
    return result


def flow_error(F_gt, F_est):
    # Remember: Flow vector = (u,v)

    # Compute error
    E_du = F_gt[:, :, 0] - F_est[:, :, 0]
    E_dv = F_gt[:, :, 1] - F_est[:, :, 1]
    E = np.sqrt(E_du ** 2 + E_dv ** 2)

    # Set the error of the non valid (occluded) pixels to 0
    F_gt_val = F_gt[:, :, 2]
    E[F_gt_val == 0] = 0

    MSE = np.mean(E[F_gt_val != 0])
    PEPN = np.sum(E[F_gt_val != 0] > 3) * 100. / len(E[F_gt_val != 0])

    print('MSE: ' + str(MSE))
    print('PEPN: ' + str(PEPN))

    return MSE, PEPN


def flow_visualization(u, v, dil_size=0):
    H = u.shape[0]
    W = u.shape[1]
    hsv = np.zeros((H, W, 3))

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(u, v)

    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # hsv[:,:,2] = ((mag - np.min(mag))/np.max(mag))*255
    hsv[:, :, 1] = 255

    hsv[:, :, 0] = ndimage.grey_dilation(hsv[:, :, 0], size=(dil_size, dil_size))
    hsv[:, :, 2] = ndimage.grey_dilation(hsv[:, :, 2], size=(dil_size, dil_size))

    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype=np.uint8)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


def flow_read(filename):
    # loads flow field F from png file

    # Read the image
    # -1 because the values are not limited to 255
    # OpenCV reads in BGR order of channels
    I = cv2.imread(filename, -1)

    # Representation:
    #   Vector of flow (u,v)
    #   Boolean if pixel has a valid value (is not an occlusion)
    F_u = (I[:, :, 2] - 2. ** 15) / 64
    F_v = (I[:, :, 1] - 2. ** 15) / 64
    F_valid = I[:, :, 0]

    # Matrix with vector (u,v) in the channel 0 and 1 and boolean valid in channel 2
    return np.transpose(np.array([F_u, F_v, F_valid]), axes=[1, 2, 0])
