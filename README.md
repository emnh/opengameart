# Categorizing OpenGameArt (using Grid t-SNE)

https://opengameart.org/ is a page for sharing reusable art for computer games.

In order to make it more easily browseable I had the idea to categorize the images using machine learning.

The idea is to create a huge tile map where similar images appear together. In the future there will be links back to OGA.

Here is a similar tool: https://ml4a.github.io/guides/ImageTSNELive/ but I didn't try it yet as it seemed fiddly to get to work regarding dependencies.

If you want to reproduce my results, this tool looks more finished taking commandline arguments instead of hardcoded parameters and so on (but I also didn't try it, although I took some source code from it): https://github.com/prabodhhere/tsne-grid .

Perhaps I will create a image similarity search engine as well. It should be easy to do when already given the feature vectors of the images.

# Results

Here follows some previews of the result of my labour:

![grid tsne crop result](https://emh.lart.no/publish/opengameart/gridtsne-crop2.png "Grid TSNE Cropped Preview")

![grid tsne scaled result](https://emh.lart.no/publish/opengameart/gridtsne-small.png "Grid TSNE Scaled Preview")

Go to home page for full scale result (50 MiB image: 6400x6400, containing almost 10k 64x64 images).

Again, sorry that there are no links back to OGA yet, and that big tile sheets are just smushed when scaled to 64x64.

Home page: https://emh.lart.no/publish/opengameart/

Source code is at "src". Sorry for lots of hard coded details.

# Progress
 - DONE: Mirror OpenGameArt
 - DONE: Unpack all zip files (but not processed yet)
 - TODO: Unpack all sprites (done for /files, not done for /unpacked. worth it? most pack files contain individuals)
 - DONE: Get feature vector of each image (src/predict.py)
 - DONE: Run t-SNE to get 2D embedding (src/tsne.py)
 - DONE: Test it with primitive HTML (src/html-out.py)
 - DONE: Run lapjv (src/grid.py) (https://github.com/src-d/lapjv) on t-SNE to embed on grid
 - TODO: Run on RTX 2080 Ti with enough memory to process all images. Perhaps do the full 4-500k image set?
 - TODO: Some way to link back to the original OpenGameArt page from tiled grid t-SNE, so you can get the art you see that you want.
 - TODO: Set transparent background color to black, both in predict.py and grid.py.
 - TODO: Add command line arguments, add scripts to run.
 - TODO: Move data alongside image files, not in dev folder.

# Future
 - Perhaps look into UMAP:
  - https://arxiv.org/abs/1802.03426
  - https://github.com/lmcinnes/umap

# Software

I am using WSL 2 on Windows. It was a pain to install all the software. 
First had to get experiment dev channel Windows.
Then install NVIDIA cuda driver.
I tried with docker, and it worked, but the non-persistent environment bugged me.
Then I was mixing and matching tensorflow packages in pip, but didn't work with cuda 11 which was the easiest to get from the web page.
So in the end I went with anaconda. It's really slow to compute compatible packages, but it works well.

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
./Anaconda3-2020.07-Linux-x86_64.sh
conda init
conda install tensorflow
conda install cudnn
conda install tensorflow-gpu
# DEPRECATED: conda install -c rapidsai-nightly -c nvidia -c conda-forge -c defaults rapids=0.15 python=3.8 cudatoolkit=10.1
# NEW: conda create -n rapids-core-0.17 -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.17 python=3.8 cudatoolkit=11.0
# follow https://developer.nvidia.com/blog/announcing-cuda-on-windows-subsystem-for-linux-2/

docker pull rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04
```
