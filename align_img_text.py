import cv2
import numpy as np

src = 255 -  cv2.imread('crop335_382_192_325_pBg.jpg', 0)
scores = []
image_main = cv2.imread('crop335_382_192_325_pBg.jpg')
# w = image_main.shape[1]
# h = image_main.shape[0]

# get width and height
height, width = image_main.shape[:2]
# display width and height
print("The height of the image is: ", height)
print("The width of the image is: ", width)

# small_dimention = min(h,w)
# src = src[:small_dimention, :small_dimention]

out = cv2.VideoWriter('rotate.avi',
                      cv2.VideoWriter_fourcc('M','J','P','G'),
                      15, (width,width))

def rotate(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def sum_rows(img):
    # Create a list to store the row sums
    row_sums = []
    # Iterate through the rows
    for r in range(img.shape[0]-1):
        # Sum the row
        row_sum = sum(sum(img[r:r+1,:]))
        # Add the sum to the list
        row_sums.append(row_sum)
    # Normalize range to (0,255)
    row_sums = (row_sums/max(row_sums)) * 255
    # Return
    return row_sums

def display_data(roi, row_sums, buffer):    
    # Create background to draw transform on
    bg = np.zeros((buffer*2, buffer*2), np.uint8)    
    # Iterate through the rows and draw on the background
    for row in range(roi.shape[0]-1):
        row_sum = row_sums[row]
        bg[row:row+1, :] = row_sum
    left_side = int(buffer/3)
    bg[:, left_side:] = roi[:,left_side:]   
    cv2.imshow('bg1', bg)
    k = cv2.waitKey(1)
    # out.write(cv2.cvtColor(cv2.resize(bg, (320,320)), cv2.COLOR_GRAY2BGR))
    out.write(cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR))
    return k

# Rotate the image around in a circle
angle = 0
while angle <= 360:
    # Rotate the source image
    img = rotate(src, angle)    
    # Crop the center 1/3rd of the image (roi is filled with text)
    h,w = img.shape
    buffer = min(h, w) - int(min(h,w)/1.5)
    #roi = img.copy()
    roi = img[int(h/2-buffer):int(h/2+buffer), int(w/2-buffer):int(w/2+buffer)]
    # Create background to draw transform on
    bg = np.zeros((buffer*2, buffer*2), np.uint8)
    # Threshold image
    _, roi = cv2.threshold(roi, 140, 255, cv2.THRESH_BINARY)
    # Compute the sums of the rows
    row_sums = sum_rows(roi)
    # High score --> Zebra stripes
    score = np.count_nonzero(row_sums)
    if sum(row_sums) < 100000: scores.append(angle)
    k = display_data(roi, row_sums, buffer)
    if k == 27: break
    # Increment angle and try again
    angle += .5
cv2.destroyAllWindows()

# Create images for display purposes	
display = src.copy()
# Create an image that contains bins. 
bins_image = np.zeros_like(display)
for angle in scores:
    # Rotate the image and draw a line on it
    display = rotate(display, angle)    
    cv2.line(display, (0,int(h/2)), (w,int(h/2)), 255, 1)
    display = rotate(display, -angle)
    # Rotate the bins image
    bins_image = rotate(bins_image, angle)
    # Draw a line on a temporary image
    temp = np.zeros_like(bins_image)
    cv2.line(temp, (0,int(h/2)), (w,int(h/2)), 50, 1)
    # 'Fill' up the bins
    bins_image += temp
    bins_image = rotate(bins_image, -angle)

# Find the most filled bin
for col in range(bins_image.shape[0]-1):
	column = bins_image[:, col:col+1]
	if np.amax(column) == np.amax(bins_image): x = col
for col in range(bins_image.shape[0]-1):
	column = bins_image[:, col:col+1]
	if np.amax(column) == np.amax(bins_image): y = col
# Draw circles showing the most filled bin
cv2.circle(display, (x,y), 560, 255, 5)
