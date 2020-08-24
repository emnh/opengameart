#!/usr/bin/env python3

import numpy as np
import cv2 as cv
#filename = '/mnt/d/opengameart/files/grass-tiles-2-small.png'
#filename = '/mnt/d/opengameart/unpacked/Atlas_0.zip/Atlas_0/terrain_atlas.png'
filename = 'test.png'

img = cv.imread(filename)
#cv.imwrite("test.png", img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.

# CORNERS
#img[dst>0.01*dst.max()]=[0,0,255]
#cv.imwrite("corners.png", img)

#cv.imshow('dst',img)
#if cv.waitKey(0) & 0xff == 27:
#    cv.destroyAllWindows()

#print(dir(cv))
surf = cv.xfeatures2d.SURF_create(400)
#kp, des = surf.detectAndCompute(img,None)
#surf.setHessianThreshold(50000)
kp, des = surf.detectAndCompute(img,None)
print(len(kp))
img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
cv.imwrite("blobs.png", img2)
