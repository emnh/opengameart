#!/usr/bin/env python3

import numpy as np
import cv2 as cv
#from PIL import Image
#filename = '/mnt/d/opengameart/files/grass-tiles-2-small.png'
#filename = '/mnt/d/opengameart/unpacked/Atlas_0.zip/Atlas_0/terrain_atlas.png'
#filename = '/mnt/d/opengameart/files/terrain2_6.png'
#filename = '/mnt/d/opengameart/files/Grasstop.png'
filename = '192x224.png'

#Image.open(filename).convert('RGBA').save('test.png')

img = cv.imread(filename)

gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
high_thresh, thresh_im = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
lowThresh = 0.5*high_thresh

edges = cv.Canny(img, lowThresh, high_thresh)
cv.imwrite('canny.png', edges)

#cv.imwrite("test.png", img)
#gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
#gray = np.float32(gray)
#dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
#dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.

# CORNERS
#img[dst>0.01*dst.max()]=[0,0,255]
#cv.imwrite("corners.png", img)

#cv.imshow('dst',img)
#if cv.waitKey(0) & 0xff == 27:
#    cv.destroyAllWindows()

#print(dir(cv))
#surf = cv.xfeatures2d.SURF_create(400)
#kp, des = surf.detectAndCompute(img,None)
#surf.setHessianThreshold(100)
#kp, des = surf.detectAndCompute(img,None)
#print(len(kp))
#img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
#cv.imwrite("blobs.png", img2)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
fast.setThreshold(30)
fast.setNonmaxSuppression(True)
fast.setType(cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
kp = fast.detect(img, None)
#print(kp[0].size)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

d = {}
for i, k in enumerate(kp):
    if i - 1 >= 0 and k.pt[1] == kp[i - 1].pt[1]:
        dx = k.pt[0] - kp[i - 1].pt[0]
        if dx in d:
            d[dx] += 1
        else:
            d[dx] = 1
        #print(dx, k.pt[1])
#for k, v in sorted(d.items(), key=lambda x: x[1]):
#    print(k, v)
        # Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv.imwrite('fast_true.png', img2)