import cv2


def ResizeImage(image, width=700):
    return cv2.resize(image, (width, int(image.shape[0] * (width / image.shape[1]))))