#!/usr/bin/env python3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.models import Model
import numpy as np
import time
import os

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

if len(sys.argv) > 1:
    path = sys.argv[1]
    for fname in os.listdir(path):
        img_path = os.path.join(path, fname)
        #img_path = 'elephant.jpg'
        x = loadImage(img_path)
        start = time.time()
        #predictions = model.predict(x)
        #for _, pred, prob in decode_predictions(predictions)[0]:
        #    print("predicted %s with probability %0.3f" % (pred, prob))
        features = feat_extractor.predict(x)
        end = time.time()
        print(features)
        print(features.shape)
        print(str(end - start) + " s")
        break
