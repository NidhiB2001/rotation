# https://stackoverflow.com/questions/57713358/how-to-rotate-skewed-fingerprint-image-to-vertical-upright-position


import cv2
import numpy as np


img_name = 'crop_roy.jpg' 
image = cv2.imread('input_img/'+img_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print('gray1.......\n', gray)
gray = 255 - gray
# print('255-gray..........\n', gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# print('thresholding::::::::::::\n',thresh)
# cv2.imwrite('thresh_'+img_name, thresh)

# Compute rotated bounding box
coords = np.column_stack(np.where(thresh > 0))
# print('coordinates"""""""""""\n', coords)
# cv2.imwrite('box_'+img_name, coords)
angle1 = cv2.minAreaRect(coords)[-1]
print('angle1>>>>>>>>>>>>>>', angle1)           #  'crop56.jpg' = 3.791165828704834

if 1 < -45:
    angle = -(90 + angle1)
elif angle1==90 or angle1==0 or angle1==-0:
    angle = 20
else:
    angle = -angle1
print('rotated angle<<<<<<', angle)

# Rotate image to deskew
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

cv2.imwrite('rotated'+str(angle1)+img_name, rotated)
