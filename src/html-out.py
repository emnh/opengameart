import json
import numpy as np

embeddings = json.loads(open('embeddings.txt').read())

data = []
lines = open('list.txt').readlines()
outfd = open('tsne.html', 'w')
for line, embedding in zip(lines, embeddings):
    d = json.loads(line.rstrip())
    data.append(d["features"])
    path = d["path"]
    path = path.replace("/emh-dev", "file://d:/dev")
    top = embedding[0] * 600 + 1600
    left = embedding[1] * 600 + 1600
    outfd.write(f"<img src=\"{path}\" style=\"position: absolute; left: {left}; top: {top};\"></img>")
outfd.close()