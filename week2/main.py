import cv2

from week2.functions.gaussianModel import OneGaussianVideo

if __name__ == '__main__':
    # Task 0, read data, train model and initialize
    modelG = OneGaussianVideo()  # Reads video, initializes class
    mean_img = modelG.modeltrainGaussian()  # Compute gaussian parameters (mean img, etc)
    # cv2.imshow('', mean_img)
    cv2.waitKey()

    # Task 1
    print('___________Task 1___________')
    modelG.classifyTest(alpha=10, rho=0.2, isAdaptive=False, showVideo=True)  # Apply gaussian to frames
    cv2.destroyAllWindows()

    # TODO  Evaluation of task 1, mAP

    # Task 2
    print('___________Task 2___________')
    modelG.classifyTest(alpha=3, rho=0.2, isAdaptive=True, showVideo=True)  # Apply gaussian to frames
    cv2.destroyAllWindows()

    # TODO Evaluation of task 2, mAP

    # Task 3
    print('___________Task 3___________')
    modelG.state_of_art()
    cv2.destroyAllWindows()

    # TODO Evaluation of task 3, mAP

    # TODO Task 4

    print('Competed!')
