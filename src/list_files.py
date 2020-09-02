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
    unpackedSpritesDir = '/mnt/i/opengameart/unpacked_sprites'
    
    spritePaths = getFiles(spriteDir, 'sprites.txt')
    filePaths = getFiles(filesDir, 'files.txt')
    unpackedPaths = getFiles(unpackedDir, 'unpacked.txt')
    unpackedSpritesPaths = getFiles(unpackedSpritesDir, 'unpacked_sprites.txt')

    # Determine which images have been split to sprites
    l = len('_00000.png')
    usedPaths = [(x[:-l] + '.png').replace(spriteDir, filesDir) for x in spritePaths]
    d = {}
    for path in usedPaths:
        d[path] = True

    # Determine which images have been split to sprites
    usedPaths2 = [(x[:-l] + '.png').replace(unpackedSpritesDir, unpackedDir) for x in spritePaths]
    d2 = {}
    for path in usedPaths2:
        d2[path] = True

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

    count2 = 0
    noncount2 = 0
    paths2 = []
    for path in unpackedPaths:
        if path in d2:
            noncount2 += 1
        else:
            count2 += 1
            paths2.append(path)
    print("non-spritesheets2", count2, "spritesheets2", noncount2)

    paths.extend(paths2)
    paths.extend(spritePaths)
    paths.extend(unpackedPaths)
    paths.sort()

    blacklisted = open('blacklist-expanded.txt').readlines()
    blackpaths = []
    for path in blacklisted:
        path = path.rstrip()
        fpath = path.replace('files/', filesDir + '/')
        upath = path.replace('files/', unpackedDir + '/')
        spath = path.replace('files/', spriteDir + '/')
        spath = os.path.splitext(spath)[0]
        blackpaths.append(fpath)
        blackpaths.append(upath)
        blackpaths.append(spath)
    newpaths = []
    bcount = 0
    for path in paths:
        black = False
        for bpath in blackpaths:
            if path.startswith(bpath):
                #print("blacklisting file", path, " it starts with ", bpath)
                if not black:
                    bcount += 1
                black = True
            else:
                pass
                #print(path, bpath)
        if not black:
            newpaths.append(path)
    print("blacklisted", bcount)
    paths = newpaths
    for i, path in enumerate(paths):
        paths[i] = path + '.np\0'

    bigsavefile = 'allpaths.txt'
    fd = open(bigsavefile, 'w')
    fd.writelines(paths)
    fd.close()
