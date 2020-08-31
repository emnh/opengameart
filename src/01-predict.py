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
import math
import multiprocessing
from glob import glob
from PIL import Image

#model = VGG16(weights='imagenet', include_top=False)
#model = VGG16(weights='imagenet', include_top=True)
model = VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

def loadImage(args):
    outpath, img_path = args
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    except:
        print("ERROR LOADING IMAGE: " + img_path)
        return None

def processFiles():
    #model.summary()
    #feat_extractor.summary()

    #if len(sys.argv) > 1:
    rootpath = '/mnt/i/opengameart/files'
    #prevfd = open('files-list.txt')
    #prevdata = prevfd.read()
    #d = {}
    #for x in prevdata:
    #    d[x] = True
    #prevdata = d
    #prevfd.close()
    #outfd = open('files-list.txt', 'a')
    start = time.time()
    i = 0
    batch = []
    #for fname in os.listdir(path):
    paths = []
    for path, dirs, files in os.walk(rootpath, topdown=False):
        for fname in files:
            flow = fname.lower()
            if not (flow.endswith('.png') or flow.endswith('.jpg')):
                continue
            img_path = os.path.join(path, fname)
            paths.append(img_path)
    paths.sort()

    batchSize = 16
    with multiprocessing.Pool(batchSize) as p:
        for img_path in paths:
            #if "\"" + img_path + "\"" in prevdata:
                #print("Already got "+ img_path)
            #    continue
            outpath = img_path + '.np'
            if os.path.exists(outpath):
                continue
            #predictions = model.predict(x)
            #for _, pred, prob in decode_predictions(predictions)[0]:
            #    print("predicted %s with probability %0.3f" % (pred, prob))
            batch.append([outpath, img_path])
            i += 1
            if i > len(files) or len(batch) >= batchSize:
                #xs = np.shape((len(batch)))
                #print(x.shape)
                images = p.map(loadImage, batch)
                xs = np.zeros((len(batch), 224, 224, 3))
                for j, x in enumerate(images):
                    xs[j] = x[0]
                features = feat_extractor.predict(xs)
                #flist = features[0].tolist()
                end = time.time()
                #d = {
                #    "path": img_path,
                #    "features": flist
                #}
                #outfd.write(json.dumps(d) + "\n")
                #print(features.shape)
                for (outpath2, _), feature, image in zip(batch, features, images):
                    #print(".", end="")
                    if image != None:
                        fd = open(outpath2, 'wb')
                        # TODO: what is the difference between ravel and flatten?
                        fd.write(feature.ravel().tobytes())
                        fd.close()
                    #print(features.shape)
                #print("")
                #if i % 10 == 0:
                print(str((end - start) / i) + " s per image. " + str(i) + " images: file " + outpath)
                batch = []

def prepImage(args):
    k, imgArray = args
    img = Image.fromarray(imgArray)
    x = np.array(img.resize((224, 224)).convert('RGB'))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def processFiles2():
    sys.path += ['/mnt/d/dev/opengameart/bhtsne']
    import grid

    image_np_pattern = '/mnt/d/opengameart/sprites/*.np'
    out_res = 32
    out_dim = math.ceil(math.sqrt(len(glob(image_np_pattern))))
    to_plot = np.square(out_dim)
    #to_plot = 100

    img_collection = grid.readImages(image_np_pattern, out_res, to_plot)[0:to_plot]
    imageColors = grid.getImageColors(img_collection)
    bigImage = grid.prepareImages(img_collection, out_dim, out_res)
    start = time.time()
    out = np.zeros((to_plot, 4096), np.float32)
    batchSize = 18
    batch = []
    with multiprocessing.Pool(batchSize) as p:
        for k in range(to_plot):
            x, y = k % out_dim, k // out_dim
            h_range = x * out_res
            w_range = y * out_res
            imgArray = bigImage[h_range:h_range + out_res, w_range:w_range + out_res, :]
            #print(features[0])
            end = time.time()
            batch.append([k, imgArray])
            if k + 1 >= to_plot or len(batch) >= batchSize:
                images = p.map(prepImage, batch)
                xs = np.zeros((len(batch), 224, 224, 3))
                for j, x in enumerate(images):
                    xs[j] = x[0]
                features = feat_extractor.predict(xs)
                #print(features.shape)
                for (k2, _), i in zip(batch, range(features.shape[0])):
                    out[k2] = features[i]
                batch = []
                print(str((end - start) / (k + 1)) + " s per image. " + str(k) + " images: file " + img_collection[k])
    fd = open('predict2.np', 'wb')
    fd.write(out.flatten().tobytes())
    fd.close()

if __name__ == '__main__':
    processFiles()
    #processFiles2()
