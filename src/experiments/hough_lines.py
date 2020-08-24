#!/usr/bin/env python3
"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def main(argv):
    default_file = '192x224.png'
    default_file = '/mnt/d/opengameart/files/Grasstop.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    #gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_UNCHANGED)
    src = cv.cvtColor(src, cv.COLOR_BGRA2GRAY)
    cv.imwrite('gray.png', src)
    #src = cv.GaussianBlur(src, (3, 3), 1)
    x = 1
    y = 0
    #dst = cv.Sobel(src, cv.CV_8UC1, x, y, ksize = 15)
    dst = cv.Sobel(src, cv.CV_8UC1, x, y, ksize=3)

    #dst = cv.cvtColor(dst, cv.COLOR_RGBA2GRAY)
    cv.imwrite('sobel.png', dst)
    #dst = abs(255 - np.uint8(dst))
    if x > 0:
        pass
        #dst = abs(255 - dst)

    #dst = cv.Canny(src, 300, 400, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    threshold = 250
    done = False
    while not done:
        lines = cv.HoughLines(dst, 1, np.pi / 180, threshold, None, 0, 0)
        lines2 = []
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                # print(theta)
                a = math.cos(theta)
                b = math.sin(theta)
                if y > 0 and abs(a) <= 1.0e-6 or x > 0 and abs(b) <= 1.0e-6:
                    lines2.append(lines[i])
        if 4 <= len(lines2) <= 100 or threshold <= 10:
            done = True
        else:
            threshold -= 10


    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            #print(theta)
            a = math.cos(theta)
            b = math.sin(theta)
            if y > 0 and abs(a) <= 1.0e-6 or x > 0 and abs(b) <= 1.0e-6:
                #print(lines[i])
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, threshold, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    #cv.imshow("Source", src)
    #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    cv.imwrite('hlines.png', cdst)
    cv.imwrite('hlinesP.png', cdstP)

    #cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
