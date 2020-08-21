# Categorizing OpenGameArt

The idea is to create a huge tile map where similar images appear together. In the future there will be links back to OGA.

Perhaps I will create a image similarity search engine as well. It should be easy to do when already given the feature vectors of the images.

![grid tsne crop result](https://emh.lart.no/publish/opengameart/gridtsne-crop2.png "Grid TSNE Cropped Preview")

![grid tsne scaled result](https://emh.lart.no/publish/opengameart/gridtsne-small.png "Grid TSNE Scaled Preview") Go to h

Home page: https://emh.lart.no/publish/opengameart/

 - DONE: Mirror OpenGameArt
 - DONE: Unpack all zip files (but not processed yet)
 - TODO: Unpack all sprites
 - DONE: Get feature vector of each image (src/predict.py)
 - DONE: Run t-SNE to get 2D embedding (src/tsne.py)
 - DONE: Test it with primitive HTML (src/html-out.py)
 - DONE: Run lapjv (src/grid.py) (https://github.com/src-d/lapjv) on t-SNE to embed on grid
 - TODO: Some way to link back to the original OpenGameArt page from tiled grid t-SNE, so you can get the art you see that you want.
