#!/usr/bin/env python3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.compiler import xla
#from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from keras.applications.vgg16 import decode_predictions, preprocess_input
#from keras.models import Model
#from tensorflow.compiler import xla
import numpy as np
import time
import os
import sys
import PIL
import json

#model = VGG16(weights='imagenet', include_top=False)
#model = VGG16(weights='imagenet', include_top=True)
model = VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

def loadImage(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

#model.summary()
#feat_extractor.summary()

#if len(sys.argv) > 1:
path = '/mnt/d/opengameart/files'
prevfd = open('files-list.txt')
prevdata = prevfd.read()
prevfd.close()
outfd = open('files-list.txt', 'a')
start = time.time()
i = 0
for fname in os.listdir(path):
    flow = fname.lower()
    if not (flow.endswith('.png') or flow.endswith('.jpg')):
        continue
    img_path = os.path.join(path, fname)
    if "\"" + img_path + "\"" in prevdata:
        print("Already got "+ img_path)
        continue
    i += 1
    try:
        x = loadImage(img_path)
    except:
        print("ERROR LOADING IMAGE: " + fname)
        continue
    #predictions = model.predict(x)
    #for _, pred, prob in decode_predictions(predictions)[0]:
    #    print("predicted %s with probability %0.3f" % (pred, prob))
    features = feat_extractor.predict(x)
    flist = features[0].tolist()
    end = time.time()
    d = {
        "path": img_path,
        "features": flist
    }
    outfd.write(json.dumps(d) + "\n")
    #print(features.shape)
    if i % 10 == 0:
        print(str((end - start) / i) + " s per image. " + str(i) + " images: file " + fname)
outfd.close()