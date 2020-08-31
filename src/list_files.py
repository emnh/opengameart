#!/usr/bin/env python3

import os
import sys

def getFiles(imageDir, savefile):
    if not os.path.exists(savefile):
        paths = []
        for root, dirs, files in os.walk(imageDir):
            for file in files:
                if file.lower().endswith('.np'):
                    file2 = file[:-len('.np')]
                    paths.append(os.path.join(root, file2))
        fd = open(savefile, 'w')
        fd.writelines([x + '\n' for x in paths])
        fd.close()
    else:
        fd = open(savefile, 'r')
        paths = [x.rstrip() for x in fd.readlines()]
        fd.close()
    return paths

if __name__ == '__main__':
    spriteDir = '/mnt/i/opengameart/sprites'
    filesDir = '/mnt/i/opengameart/files'
    unpackedDir = '/mnt/i/opengameart/unpacked'
    
    spritePaths = getFiles(spriteDir, 'sprites.txt')
    filePaths = getFiles(filesDir, 'files.txt')
    unpackedPaths = getFiles(unpackedDir, 'unpacked.txt')

    # Determine which images have been split to sprites
    l = len('_00000.png')
    usedPaths = [(x[:-l] + '.png').replace(spriteDir, filesDir) for x in spritePaths]
    d = {}
    for path in usedPaths:
        d[path] = True

    count = 0
    noncount = 0
    paths = []
    for path in filePaths:
        if path in d:
            noncount += 1
        else:
            count += 1
            paths.append(path)
    print("non-spritesheets", count, "spritesheets", noncount)

    paths.extend(spritePaths)
    paths.extend(unpackedPaths)
    paths.sort()

    bigsavefile = 'allpaths.txt'
    fd = open(bigsavefile, 'w')
    fd.writelines([x + '\n' for x in paths])
    fd.close()
