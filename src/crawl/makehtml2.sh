#!/bin/bash
#s3cmd la --recursive | fgrep /files/ > out2.txt
IMAGES=$(
grep -o 's3://.*' out2.txt |
grep '.png$' |
sed -s 's@s3://opengameart/@@' |
sed 's/^/"/' | sed 's/$/",/' |
fgrep -v 1bit
)
#sed -s 's@s3://opengameart/@https://opengameart.fra1.cdn.digitaloceanspaces.com@' |
LINKS=$(cat links.js)
#IMAGES=$(cat ogapng.txt | sed 's/^/"/' | sed 's/$/",/')
cat << EOF > ~/public_html/publish/opengameart/randomData.js
const images = [
  $IMAGES
];
const links = $LINKS;
EOF
cat << EOF > ~/public_html/publish/opengameart/random.html
<html>
<head>
<style>
img {
  width: auto;
  height: auto;
  max-width: 10%;
  max-height: 200px;
  display: inline;
  top: 0px;
  border: 2px solid black;
  //width: 9%;
  //max-height: 200px;
}
</style>
<script type="text/javascript" src="randomData.js"></script>
</head>
<body>
  <script type='text/javascript'>
    const speed = 3;
    const files = {};

    var k = 0;
    for (var i in images) {
      var uri = images[i];
      if (uri.startsWith('files/')) {
        files[uri] = [uri];
      } else {
        var uri2 = uri.split('/').slice(0, 2).join('/');
        if (uri2 in files) {
          files[uri2].push(uri);
        } else {
          files[uri2] = [uri];
        }
        k++;
        if (k < 10) {
          console.log(uri2);
        }
      }
    }

    const keys = Object.keys(files);

    let imgs = [];
    function addImage() {
      var r = Math.floor(Math.random() * keys.length);
      var r2 = Math.floor(Math.random() * files[keys[r]].length);
      var uri = files[keys[r]][r2];

      var img = document.createElement('img');
      var a = document.createElement('a');

      var fullURI = 'https://opengameart.org/sites/default/' + uri;
      let href = '';
      if (uri.startsWith('files/')) {
        href = links[fullURI];
        console.log('files', uri, fullURI, href);
      } else {
        let f = fullURI.replace('/unpacked/', '/files/');
        let f2 = f.split('/').slice(0, 7).join('/');
        href = links[f2];
        console.log('packed', f2, href);
      }
      a.href = 'https://opengameart.org/' + href.replace(/.html$/, '');
      document.body.append(a);
      a.append(img);
      img.src = 'https://opengameart.fra1.cdn.digitaloceanspaces.com/' + encodeURI(uri).replace(/#/g, '%23');
    }
    function scroll() {
      var doc = document.documentElement;
      //var left = (window.pageXOffset || doc.scrollLeft) - (doc.clientLeft || 0);
      var top = (window.pageYOffset || doc.scrollTop)  - (doc.clientTop || 0);
      var dist = document.body.scrollHeight - top - window.innerHeight;
      //console.log("dist", dist);
      if (dist < window.innerHeight) {
        window.scrollTo(0, document.body.scrollHeight);
        return true;
      }
      return false;
    }
    function addImages(k) {
      for (var i = 0; i < k; i++) {
        addImage();
      }
    }
    addImages(10);
    setInterval(() => scroll() && addImages(1), 100);
    setInterval(scroll, 100);
  </script>
</body>
</html>
EOF
