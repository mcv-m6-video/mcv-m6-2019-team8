import os

import cv2
import numpy as np

from PATHS import PATH_OF_IMAGES, PATH_OPTICAL_FLOW_GT
from week4.functions.gunner_farneback import gunner_farneback, evaluate_gunner_farneback

nsequence = '000045'
F_gt = os.path.join(PATH_OPTICAL_FLOW_GT, nsequence + '_10.png')
prvs = os.path.join(os.path.join(PATH_OF_IMAGES, nsequence + '_10.png'))
curr = os.path.join(os.path.join(PATH_OF_IMAGES, nsequence + '_11.png'))
names = [F_gt, prvs, curr]

best_mmse, best_pepn = evaluate_gunner_farneback(names)
