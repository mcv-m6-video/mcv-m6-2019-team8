import os
import cv2
import numpy as np
import math as mt

from PATHS import PATH_FRAMES, PATH_WEEK_2_FRAMES
from week2.modelv2.bbox import BBox


class OneGaussianVideo:
    train_frames = list
    test_frames = list
    dir_path = str
    mean_train: float
    std_train: float

    def __init__(self, train_frames=[], test_frames=[],
                 dir_path=PATH_FRAMES, mean_train=0,
                 std_train=0):
        self.dir_path = dir_path
        self.train_frames = []
        self.test_frames = []
        self.gaussian_frames = []
        self.state_art_test_frames = []
        self.mean_train = []
        self.sstd_train = []
        self.readVideoBW(self.dir_path)

    def readVideoBW(self, dir_path=PATH_WEEK_2_FRAMES):
        # Frame path
        frame_path = 'D:/m6-mmalak/datasets/AICity_data/AICity_data/train/S03/c010/' + 'frames'  # /frames -> For the full data
        # gt path
        gt_path = 'D:/m6-mmalak/datasets/AICity_data/AICity_data/train/S03/c010/gt/'
        frame_list = sorted(os.listdir(frame_path))
        num_frames = len(frame_list)

        j = 0
        for i, j in enumerate(frame_list):
            # print('Reading Frame: ' + str(i))
            if i <= mt.trunc(0.25 * num_frames):  # 25% frames for training, 75% for testing
                image_path = frame_path + '/image' + str(i) + '.jpg'
                im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale only
                if im is not None:
                    im = cv2.resize(im, (0, 0), fx=0.3, fy=0.3)  # Make images smaller
                    # im_v=np.reshape(im,im.shape[0]*im.shape[1])
                    self.train_frames.append([i, im])
                    print('Reading Training Frame: ' + str(i))
            else:
                image_path = frame_path + '/image' + str(i) + '.jpg'
                im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale only
                if im is not None:
                    im = cv2.resize(im, (0, 0), fx=0.3, fy=0.3)  # # Make images smaller
                    # im_v = np.reshape(im, im.shape[0] * im.shape[1])
                    self.test_frames.append([i, im])
                    print('Reading Testing Frame: ' + str(i))

    # def creategt(self):

    def modeltrainGaussian(self):
        # Get the mean and std of training frames, used for background substraction
        print('Training Gaussian model')
        t_frames = []
        for i, frame in self.train_frames:
            t_frames.append(frame)
        self.mean_train = np.mean(t_frames, 0).astype(np.uint8)
        self.std_train = np.std(t_frames, 0).astype(np.uint8)
        #        cv2.imshow('',self.mean_train)
        #        cv2.waitKey(20)
        return self.mean_train

    def classifyTest(self, alpha, rho, isAdaptive, showVideo):
        print('Classifying frames')
        for i, frame in self.test_frames:
            out_frame = np.empty(np.shape(frame))  # Initialize empty frame # Initialize empty frame
            if isAdaptive:
                background = frame != 255  # Only background pixels
                # Equations from the slides [Adaptive Modelling]
                self.mean_train[background] = rho * frame[background] + (1 - rho) * self.mean_train[background]
                self.std_train[background] = np.sqrt(
                    rho * (frame[background] - self.mean_train[background]) ** 2 + (1 - rho) * (
                            self.std_train[background] ** 2))
                out_frame = np.abs(frame - self.mean_train) >= alpha * (self.std_train + 2)
            else:
                # Equations from the slides [Gaussian Modelling]
                out_frame = np.abs(frame - self.mean_train) >= alpha * (self.std_train + 2)
            #                for row in range(np.shape(frame)[0]):
            #                    for col in range(np.shape(frame)[1]):
            #                        if np.abs(frame[row, col]-self.mean_train[row, col]) >= alpha*(self.std_train[row, col]+2):
            #                            out_frame[row, col] = 1
            #                        else:
            #                            out_frame[row, col] = 0

            # Clean image with morphological operators (noisy areas, and holes)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            out_frame = cv2.morphologyEx(out_frame.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            out_frame = cv2.morphologyEx(out_frame.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            self.gaussian_frames.append([i, out_frame.astype(np.uint8)])
            if showVideo == True:
                cv2.imshow('', out_frame.astype(np.uint8) * 255)
                cv2.waitKey(20)

    def state_of_art(self):
        print('State of the art')
        # Source: https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        for i, frame in self.test_frames:
            # print('frame: '+str(i))
            # bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(frame, learningRate=0.01)
            # cv2.imshow('frame',fgmask)
            self.state_art_test_frames.append([i, fgmask])
            cv2.imshow('', fgmask.astype(np.uint8))
            cv2.waitKey(20)

    @staticmethod
    def getgt_detections(directory_txt):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        boxes_gt = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            boxes_gt.append(BBox(frameid, topleft, width, height, confidence))

        return boxes_gt

    def creategt(self, frames):
        # Not using it
        gt_dir = self.dir_path + '/gt'
        boxes_gt = self.getgt_detections(gt_dir)
        box_frame = []
        white_image = 255 * np.ones_like(frames[0][1])

        for i in frames:
            for u in boxes_gt:
                if u.frame_id == i[0]:
                    box_frame[i].append(u)
        for i in frames:
            for u in box_frame[i]:
                [xmin, ymin, xmax, ymax] = u.to_result()
                roi = i[1][xmin:xmax, ymin:ymax]
                for val in np.unique(roi)[1:]:  # step 2
                    mask = np.uint8(roi == val)  # step 3
                    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
                    white_image[labels == largest_label] = val


"""
    @staticmethod
    def getgt_detections(directory_txt, num_frames):
       """ """Read txt files containing bounding boxes (ground truth and detections).""""""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        vid_fr = []
        frameid_saved = 1
        Boxes_list = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            Boxes_list.append(('frameid':frameid, 'topleft':topleft, 'width': width, 'height':height, 'confidence':confidence))
        for i in range(0, num_frames):
            items = [item for item in Boxes_list if item.frame_id == i]
            if items:
                vid_fr.append(Frame(i, items))
            txt_gt.close()
        return vid_fr"""
