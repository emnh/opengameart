const $ = require('jquery');
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

const table = $("<table/>");
$("body").append(table);

let imgs = [];
function addImage(r) {
  //var r = Math.floor(Math.random() * keys.length);
  //var r2 = Math.floor(Math.random() * files[keys[r]].length);
  
  for (var r2 = 0; r2 < files[keys[r]].length; r2++) {
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
    const tr = $("<tr/>");
    const td = $("<td/>");
    table.append(tr);
    tr.append(td);
    td.append(a);
    //document.body.append(a);
    a.append(img);
    img.src = 'https://opengameart.fra1.cdn.digitaloceanspaces.com/' + encodeURI(uri).replace(/#/g, '%23');
  }
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
addImage(0);
