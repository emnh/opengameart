#!/bin/bash
#s3cmd la --recursive | fgrep /files/ > out2.txt
IMAGES=$(
grep -o 's3://.*' out2.txt |
grep '.png$' |
sed -s 's@s3://opengameart@https://opengameart.fra1.cdn.digitaloceanspaces.com@' |
sed 's/^/"/' | sed 's/$/",/' |
fgrep -v 1bit
)
#IMAGES=$(cat ogapng.txt | sed 's/^/"/' | sed 's/$/",/')
cat << EOF > ~/public_html/random-pixel-art.html
<html>
<head>
<style>
img {
  width: auto;
  max-height: 90%;
  display: inline;
  position: absolute;
  top: 0px;
  //width: 9%;
  //max-height: 200px;
}
</style>
</head>
<body>
  <script type='text/javascript'>
    const speed = 3;
    const images = [
      $IMAGES
    ];
    let imgs = [];
    function addImage(oldImage) {
      var img = oldImage || document.createElement('img');
      var r = Math.floor(Math.random() * images.length);
      //document.body.insertBefore(img, document.body.firstChild);
      document.body.append(img);
      img.style.left = '0px';
      img.style.visibility = 'hidden';
      img.onload = function() {
        let maxOffset = -1000;
        for (var i = 0; i < imgs.length; i++) {
          if (imgs[i] !== img && imgs[i].complete) {
            let left = parseInt(imgs[i].style.left.replace('px', ''));
            offset = left + imgs[i].width;
            maxOffset = Math.max(offset, maxOffset);
          }
        }
        img.style.left = maxOffset + 'px';
        console.log("loaded", img.style.left);
        img.style.visibility = 'visible';
      }
      img.src = encodeURI(images[r]);
      if (oldImage === undefined) {
        imgs.push(img);
      }
    }
    function addImages(k) {
      for (var i = 0; i < k; i++) {
        addImage();
      }
    }
    addImages(10);
    //setInterval(addImage, 1000);
    function slide() {
      for (var i = 0; i < imgs.length; i++) {
        if (imgs[i].complete) {
          let left = parseInt(imgs[i].style.left.replace('px', ''));
          //console.log(left);
          let newVal = left - speed;
          imgs[i].style.left = newVal + 'px';
          if (newVal < -imgs[i].width) {
            console.log("added image");
            addImage(imgs[i]);
          }
        }
      }
      requestAnimationFrame(slide);
    }
    requestAnimationFrame(slide);
  </script>
</body>
</html>
EOF
