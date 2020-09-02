#!/usr/bin/env python3

import sys
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

k = 4

infname = sys.argv[1]
bigimg = Image.open(infname)
ar = np.array(bigimg)

w, h = bigimg.width // k, bigimg.height // k
for x in range(k):
    for y in range(k):
        outfname = infname.replace('.png', '-' + str(x) + 'x' + str(y) + '.png')
        smallimg = Image.fromarray(ar[x * w:(x + 1) * w, y * h:(y + 1) * h])
        if outfname != infname:
            print("out", outfname)
            smallimg.save(outfname)

