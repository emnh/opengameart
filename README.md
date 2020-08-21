# Categorizing OpenGameArt (using Grid t-SNE)

https://opengameart.org/ is a page for sharing reusable art for computer games.

In order to make it more easily browseable I had the idea to categorize the images using machine learning.

The idea is to create a huge tile map where similar images appear together. In the future there will be links back to OGA.

Here is a similar tool: https://ml4a.github.io/guides/ImageTSNELive/ but I didn't try it yet as it seemed fiddly to get to work regarding dependencies.

Perhaps I will create a image similarity search engine as well. It should be easy to do when already given the feature vectors of the images.

Here follows some previews of the result of my labour:

![grid tsne crop result](https://emh.lart.no/publish/opengameart/gridtsne-crop2.png "Grid TSNE Cropped Preview")

![grid tsne scaled result](https://emh.lart.no/publish/opengameart/gridtsne-small.png "Grid TSNE Scaled Preview")

Go to home page for full scale result (50 MiB image: 6400x6400, containing almost 10k 64x64 images).

Home page: https://emh.lart.no/publish/opengameart/

Source code is at "src". Sorry for lots of hard coded details.

Progress:
 - DONE: Mirror OpenGameArt
 - DONE: Unpack all zip files (but not processed yet)
 - TODO: Unpack all sprites
 - DONE: Get feature vector of each image (src/predict.py)
 - DONE: Run t-SNE to get 2D embedding (src/tsne.py)
 - DONE: Test it with primitive HTML (src/html-out.py)
 - DONE: Run lapjv (src/grid.py) (https://github.com/src-d/lapjv) on t-SNE to embed on grid
 - TODO: Some way to link back to the original OpenGameArt page from tiled grid t-SNE, so you can get the art you see that you want.
