# https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
 # https://stackoverflow.com/questions/59660933/how-to-de-skew-a-text-image-and-retrieve-the-new-bounding-box-of-that-image-pyth

import cv2
import numpy as np
from scipy.ndimage import rotate

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

if __name__ == '__main__':
    image = cv2.imread('crop322_370_509_577_pSC.jpg')
    angle, corrected = correct_skew(image)
    print('Skew angle:', angle)
    cv2.imshow('corrected', corrected)
    cv2.waitKey()