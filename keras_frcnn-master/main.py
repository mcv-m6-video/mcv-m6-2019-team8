from detector import FasterRCNNDetector
import cv2


def main():

    detector = FasterRCNNDetector(model_path='./weights/model_frcnn.hdf5')

    img = cv2.imread('images/000000.png')
    detector.detect_on_image(img, '000000.png')

    img = cv2.imread('images/000001.png')
    detector.detect_on_image(img, '000001.png')

    img = cv2.imread('images/000002.png')
    detector.detect_on_image(img, '000002.png')

    img = cv2.imread('images/000003.png')
    detector.detect_on_image(img, '000004.png')

    img = cv2.imread('images/000004.png')
    detector.detect_on_image(img, '000004.png')

    img = cv2.imread('images/000005.png')
    detector.detect_on_image(img, '000005.png')


if __name__ == '__main__':
    main()
