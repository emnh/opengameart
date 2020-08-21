import json
import numpy as np

embeddings = json.loads(open('embeddings.txt').read())
minx = 0.0
maxx = 0.0
miny = 0.0
maxy = 0.0
for embedding in embeddings:
    x, y = embedding
    minx = min(minx, x)
    maxx = max(maxx, x)
    miny = min(miny, y)
    maxy = max(maxy, y)

tw = (maxx - minx)
th = (maxy - miny)

data = []
lines = open('opengameart-files/files-list.txt').readlines()
outfd = open('tsne.html', 'w')
for line, embedding in zip(lines, embeddings):
    d = json.loads(line.rstrip())
    data.append(d["features"])
    path = d["path"]
    path = path.replace("/opengameart/files/", "file://d:/opengameart/files256/")
    x, y = embedding
    x -= minx
    y -= miny
    top = x / tw * 3200
    left = y / tw * 3200
    outfd.write(f"<img src=\"{path}\" style=\"width: 16px; max-height: 16px; position: absolute; left: {left}; top: {top};\"></img>")
outfd.close()

outfd = open('src/index.js', 'w')
outfd.write('var imgs = [')
for line, embedding in zip(lines, embeddings):
    d = json.loads(line.rstrip())
    data.append(d["features"])
    path = d["path"]
    path = path.replace("/opengameart/files/", "file://d:/opengameart/files256/")
    x, y = embedding
    x -= minx
    y -= miny
    top = x / tw * 3200
    left = y / tw * 3200
    outfd.write('"' + path + '",\n')
    #outfd.write(f"<img src=\"{path}\" style=\"width: 16px; max-height: 16px; position: absolute; left: {left}; top: {top};\"></img>")
outfd.write('];')
outfd.close()
