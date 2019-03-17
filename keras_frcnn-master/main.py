from detector import FasterRCNNDetector
import cv2


def main():

    detector = FasterRCNNDetector(model_path='./weights/model_frcnn.hdf5')

    img = cv2.imread('images/000000.png')
    detector.detect_on_image(img)



if __name__ == '__main__':
    main()
