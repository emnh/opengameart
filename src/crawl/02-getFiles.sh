#!/bin/bash
#fgrep 'File(s)' content/* |
cat content/* |
grep -o "https://opengameart.org/sites/default/files/[^'\"]*" |
fgrep -v /files/styles |
sort -u > allfiles.txt

len=$(wc -l allfiles.txt)
i=0
cat allfiles.txt | while read file; do
  i=$(($i + 1))
  echo $i / $len: $file
  bname=$(basename $file)
  path="s3://opengameart/files/$bname"
  count=`s3cmd ls "$path" | wc -l`
  if [[ $count -gt 0 ]]; then
    echo Skipping $file to $path
    sleep 0.1
    continue
  fi
  #s3cmd get s3://opengameart/files/$bname files/$bname
  wget -c $file -O files/$bname &&
  s3cmd put files/$bname s3://opengameart/files/$bname --acl-public &&
  #s3cmd setacl s3://opengameart/files/$bname --acl-public &&
  rm -f files/$bname
  sleep 2
done
