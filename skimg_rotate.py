# import cv2
# import numpy as np
# from skimage import io
# from skimage.color import rgb2gray
# from skimage.transform import rotate
# from deskew import determine_skew

# image = cv2.imread('crop322_370_509_577_pSC.jpg')
# grayscale = rgb2gray(image)
# angle = determine_skew(grayscale)
# rotated = rotate(image, angle, resize=True) * 255
# io.imsave('output.png', rotated.astype(np.uint8))


import imutils
import cv2
import numpy as np

def detect_angle(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    adaptive = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,4)

    cnts = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 45000 and area > 20:
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = mask.shape
    
    # Horizontal
    if w > h:
        left = mask[0:h, 0:0+w//2]
        right = mask[0:h, w//2:]
        left_pixels = cv2.countNonZero(left)
        right_pixels = cv2.countNonZero(right)
        return 0 if left_pixels >= right_pixels else 180
    # Vertical
    else:
        top = mask[0:h//2, 0:w]
        bottom = mask[h//2:, 0:w]
        top_pixels = cv2.countNonZero(top)
        bottom_pixels = cv2.countNonZero(bottom)
        return 90 if bottom_pixels >= top_pixels else 270

if __name__ == '__main__':
    image = cv2.imread('crop335_382_192_325_pBg.jpg')
    angle = detect_angle(image)
    Rotated_image = imutils.rotate(image, angle=-10)
    print(angle)
    
    cv2.imshow('rotateIMG', Rotated_image)
    cv2.waitKey(0)
    

