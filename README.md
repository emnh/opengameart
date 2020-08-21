# Categorizing OpenGameArt

![grid tsne result](https://emh.lart.no/publish/opengameart/gridtsne-small.png "Grid TSNE Result")

 - DONE: Mirror OpenGameArt
 - DONE: Unpack all zip files (but not processed yet)
 - TODO: Unpack all sprites
 - DONE: Get feature vector of each image (src/predict.py)
 - DONE: Run t-SNE to get 2D embedding (src/tsne.py)
 - DONE: Test it with primitive HTML (src/html-out.py)
 - DONE: Run lapjv (src/grid.py) (https://github.com/src-d/lapjv) on t-SNE to embed on grid
