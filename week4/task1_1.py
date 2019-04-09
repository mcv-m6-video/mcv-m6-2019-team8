import os
import matplotlib.pyplot as plt
import cv2
from PATHS import PATH_OPTICAL_FLOW_GT, PATH_OF_IMAGES, RESULT_DIR
from week4.functions.OpticalFlowCalculations import readOF, readOFimages, block_matching, flow_error, \
    flow_visualization, flow_read

nsequence = '000045'
sequence = []
frame = cv2.imread(os.path.join(os.path.join(PATH_OF_IMAGES, nsequence + '_10.png')))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
sequence.append(frame)
frame = cv2.imread(os.path.join(os.path.join(PATH_OF_IMAGES, nsequence + '_11.png')))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
sequence.append(frame)
F_gt = flow_read(os.path.join(PATH_OPTICAL_FLOW_GT, nsequence + '_10.png'))

<<<<<<< HEAD
flow = block_matching(sequence[0], sequence[1], block_size=(4, 4),
                      step=(8, 8), area=(4, 4), area_step=(8, 8),
=======
flow = block_matching(sequence[0], sequence[1], block_size=(24, 24), step=(8, 8), area=(24, 24), area_step=(8, 8),
>>>>>>> 2aab64a3a31d08b2c86ad819d617c25d7dad0d48
                      error_thresh=3, verbose=True)
flow_error(F_gt, flow)
rgb_flow = flow_visualization(flow[:, :, 0], flow[:, :, 1], 0)
plt.savefig(RESULT_DIR + 'rgb_flow.png')
plt.imshow(rgb_flow)
plt.show()
