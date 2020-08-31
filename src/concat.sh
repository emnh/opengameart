#!/bin/bash
#cd predict &&
#find /mnt/d/opengameart/unpacked -type f -name \*.np -print0 |
#  sort -z |
#  xargs -0 cat -- >> predict.out
cat allpaths.txt | xargs -0 cat -- >> predict.out
