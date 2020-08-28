#!/bin/sh
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -v /mnt/d:/mnt/d -w $(pwd) rapidsai/rapidsai-nightly:cuda10.1-runtime-ubuntu18.04 python3 ../src/02-tsne.py
