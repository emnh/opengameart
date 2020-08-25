#!/usr/bin/env python3

import sys
import random
import os
import tensorflow as tf
import numpy as np
import split_sheet
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from examples.tensorflow_examples.models.pix2pix import pix2pix

checkpoint_dir = 'sprite.model'
trainX = 128
trainY = 128
trainType = 'RGB'
tries = [8, 16, 24, 32, 48, 64]
#trainType = 'L'

paths = [
    ('/mnt/d/opengameart/files/ProjectUtumno_supplemental_0.png', (32, 32)),
    ('/mnt/d/opengameart/files/grass-tiles-2-small.png', (32, 32)),
    ('/mnt/d/opengameart/files/StoneBlocks_byVellidragon.png', (32, 32)),
    ('/mnt/d/opengameart/files/terrain2_6.png', (64, 64)),
    ('/mnt/d/opengameart/unpacked/Atlas_0.zip/Atlas_0/terrain_atlas.png', (32, 32)),
    ('/mnt/d/opengameart/files/Grasstop.png', (16, 16)),
    ('/mnt/d/opengameart/files/%23011-Nekogare%20hey.png', (1, 1)),
    ('/mnt/d/opengameart/files/arcadArne_sheet_org_desat.png', (16, 16)),
    ('/mnt/d/opengameart/files/Green%20Iron.png', (16, 16)),
    ('/mnt/d/opengameart/files/8x8%20pixel%20tiles_3.png', (8, 8))
    #('/mnt/d/opengameart/files/icons2_0.png', (24, 24))
]

def test():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train.shape, y_train.shape)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

def createData():

    tiles = []
    answers = []
    nonTiles = []

    for idx in range(len(paths)):
        path = paths[idx][0]
        tileWidth, tileHeight = paths[idx][1]

        if tileWidth < 4 or tileHeight < 4:
            continue

        img = Image.open(path).convert('RGBA')
        origar = np.array(img)
        #.convert('L')
        #dstH, dstV = split_sheet.getHV(img)
        #ar = np.array(dstH)
        ar = np.array(img)

        # -2 because we add a random offset to get a non-tile
        for x in range(ar.shape[0] // tileWidth - 2):
            for y in range(ar.shape[1] // tileHeight - 2):
                # Add a tile
                xo = 0
                yo = 0
                nx1 = x * tileWidth + xo
                nx2 = nx1 + tileWidth
                ny1 = y * tileHeight + yo
                ny2 = ny1 + tileHeight
                tile = ar[nx1:nx2, ny1:ny2, :]
                tp = origar[nx1:nx2, ny1:ny2, 3]
                # ignore fully transparent tiles
                if not np.equal(tp, 0).all():
                    tileImg = Image.fromarray(tile).resize((trainX, trainY)).convert(trainType)
                    tile = np.array(tileImg)
                    tiles.append(tile)
                    answers.append(1)
                    #Image.fromarray(tile).save('test.png')
                    #break
                else:
                    continue

                # Add a non-tile (wrong size)
                #for (tileWidth2, tileHeight2) in zip(tries, tries):
                if True:
                    tileWidth2 = tileHeight2 = random.choice(tries)
                    xo = 0
                    yo = 0
                    nx1 = x * tileWidth + xo
                    nx2 = nx1 + tileWidth2
                    ny1 = y * tileHeight + yo
                    ny2 = ny1 + tileHeight2
                    nonTile = ar[nx1:nx2, ny1:ny2, :]
                    tp = origar[nx1:nx2, ny1:ny2, 3]
                    if nonTile.shape == (tileWidth2, tileHeight2, ar.shape[2]) and not np.equal(tp, 0).all():
                        nonTileImg = Image.fromarray(nonTile).resize((trainX, trainY)).convert(trainType)
                        nonTile = np.array(nonTileImg)
                        nonTiles.append(nonTile)
                    else:
                        #print(nonTile.shape)
                        continue

                # Add a non-tile (tile offset by random amount)
                if True:
                    xo = random.randint(1, tileWidth - 1)
                    yo = random.randint(1, tileHeight - 1)
                    nx1 = x * tileWidth + xo
                    nx2 = nx1 + tileWidth
                    ny1 = y * tileHeight + yo
                    ny2 = ny1 + tileHeight
                    nonTile = ar[nx1:nx2, ny1:ny2]
                    tp = origar[nx1:nx2, ny1:ny2, 3]
                    if not np.equal(tp, 0).all():
                        nonTileImg = Image.fromarray(nonTile).resize((trainX, trainY)).convert(trainType)
                        nonTile = np.array(nonTileImg)
                        nonTiles.append(nonTile)
                    else:
                        continue

    #print(tiles[0].shape, answers[0].shape)

    x_train = []
    y_train = []
    indices = list(range(len(tiles) + len(nonTiles)))
    random.shuffle(indices)
    for i in indices:
        if i < len(tiles):
            x_train.append(tiles[i])
            y_train.append(answers[i])
        else:
            x_train.append(nonTiles[i  - len(tiles)])
            y_train.append(0)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    n = int(len(x_train) * 0.8)
    x_train, x_test = x_train[:n], x_train[n:]
    y_train, y_test = y_train[:n], y_train[n:]

    print("positive/negative", len(tiles), len(nonTiles))
    print("xy_train", x_train.shape, y_train.shape)

    return (x_train, y_train, x_test, y_test)

def ml(load, x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(trainX, trainY)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if not load:
        model.fit(x_train, y_train, epochs=500)
        model.save_weights(checkpoint_dir)
    else:
        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(checkpoint_dir)
    model.evaluate(x_test, y_test)

    return model

def ml2(load, x_train, y_train, x_test, y_test):
    num_classes = 2

    img_width = trainX
    img_height = trainY

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        #layers.Conv2D(64, 3, padding='same', activation='relu'),
        #layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    epochs = 5

    if not load:
        history = model.fit(
            x_train, y_train,
            # validation_data=val_ds,
            epochs=epochs
        )
        model.evaluate(x_test, y_test)
        model.save_weights(checkpoint_dir)
    else:
        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(checkpoint_dir)

    return model

def ml3(x_train, y_train, x_test, y_test):
    OUTPUT_CHANNELS = 3

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    def unet_model(output_channels):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs = 100

    #print(x_train.shape, y_train.shape)

    history = model.fit(
        (x_train, y_train),
        # validation_data=val_ds,
        epochs=epochs
    )

    model.evaluate(x_test, y_test)

    return model

def checkSize(path, model, tileWidth, tileHeight):
    tiles = []
    answers = []
    nonTiles = []

    img = Image.open(path).convert('RGBA')
    origar = np.array(img)
    ar = np.array(img)
    # Put transparency to black
    #ar = (ar[:, :, :3] * np.repeat(ar[:, :, 3][:, :, None], 3, axis=2) / 255).astype(np.uint8)

    ar = ar[0:tileWidth*10, 0:tileHeight*10, :]
    #print(ar.shape)

    # -2 because we add a random offset to get a non-tile
    for x in range(ar.shape[0] // tileWidth - 1):
        for y in range(ar.shape[1] // tileHeight - 2):
            # Add a tile
            xo = 0
            yo = 0
            nx1 = x * tileWidth + xo
            nx2 = nx1 + tileWidth
            ny1 = y * tileHeight + yo
            ny2 = ny1 + tileHeight
            tile = ar[nx1:nx2, ny1:ny2]
            tp = origar[nx1:nx2, ny1:ny2, 3]
            # ignore transparent tiles
            #if tp.all():
            tileImg = Image.fromarray(tile).resize((trainX, trainY)).convert(trainType)
            tile = np.array(tileImg)
            tiles.append(tile)
            #else:
            #    continue

            # Add a non-tile (tile offset by random amount)
            #xo = random.randint(1, tileWidth - 1)
            #yo = random.randint(1, tileHeight - 1)
            xo = tileWidth // 2
            yo = tileHeight // 2
            nx1 = x * tileWidth + xo
            nx2 = nx1 + tileWidth
            ny1 = y * tileHeight + yo
            ny2 = ny1 + tileHeight
            nonTile = ar[nx1:nx2, ny1:ny2]
            tp = origar[nx1:nx2, ny1:ny2, 3]
            #if tp.all():
            nonTileImg = Image.fromarray(nonTile).resize((trainX, trainY)).convert(trainType)
            nonTile = np.array(nonTileImg)
            nonTiles.append(nonTile)
            #else:
            #    continue

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    batch = 100
    tileProb = 0
    tileProbCount = 0
    nonTileProb = 0
    nonTileProbCount = 0
    for x in range(len(tiles) // batch + 1):
        indata = np.array(tiles[min(len(tiles), x * batch):min(len(tiles), x * batch + batch)])
        if indata.shape[0] > 0:
            predictions = probability_model.predict(indata)
            tileProb += np.sum(np.argmax(predictions, 1))
            tileProbCount += predictions.shape[0]
            print("tile batch", x, len(tiles) // batch, tileProb, tileProbCount)
    for x in range(len(nonTiles) // batch + 1):
        indata = np.array(nonTiles[min(len(nonTiles), x * batch): min(len(nonTiles), x * batch + batch)])
        if indata.shape[0] > 0:
            predictions = probability_model.predict(indata)
            nonTileProb += np.sum(np.argmax(predictions, 1))
            nonTileProbCount += predictions.shape[0]
            print("nonTile batch", x, len(nonTiles) // batch, nonTileProb, nonTileProbCount)

    #return ((tileProb - nonTileProb) / max(1, tileProbCount), nonTileProb / max(1, nonTileProbCount))
    return (tileProb / max(1, tileProbCount), nonTileProb / max(1, nonTileProbCount))
    #return (np.average(tileProbs), np.average(nonTileProbs))

def check(model, path, tileWidth, tileHeight):
    img = Image.open(path)
    data = []
    active = [x for x in tries if img.width % x == 0 and img.height % x == 0]
    if len(active) == 0:
        active = tries
    for x in active:
        tp, ntp = checkSize(path, model, x, x)
        correct = 'YES' if x == tileWidth else 'nope'
        data.append((path, correct, tileWidth, tileHeight, x, tp, ntp))
    for d in data:
        print(*d)
    #data.sort(key=lambda x: x[5])
    data.sort(key=lambda x: (1 if x[6] < x[5] else 0, x[5]))
    return data

def test(model):

    preds = []
    for idx in range(len(paths)):
        path = paths[idx][0]
        tileWidth, tileHeight = paths[idx][1]

        if tileWidth < 4 or tileHeight < 4:
            continue

        data = check(model, path, tileWidth, tileHeight)
        preds.append(("BEST PREDICTION: ", *data[-1]))
    for pred in preds:
        print(pred)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #test()
    x_train, y_train, x_test, y_test = createData()
    #ml(x_train, y_train, x_test, y_test)
    load = True
    model = ml2(load, x_train, y_train, x_test, y_test)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        data = check(model, path, 1, 1)
        print("BEST PREDICTION: ", *data[-1])
    else:
        test(model)


    #ml3(x_train, y_train, x_test, y_test)
