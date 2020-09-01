#!/bin/sh
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -v /mnt/i:/mnt/i -w $(pwd) rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04 python3 ../src/02-tsne.py
