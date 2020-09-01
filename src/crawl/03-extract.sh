#!/bin/bash
s3cmd la --recursive | fgrep -v /unpacked/ > out.txt
egrep -o 's3://.*\.((zip)|(7z)|(rar)|(tar)|(tgz)|(tar.gz)|(tar.bz2))$' out.txt |
while read file; do
  bname=$(basename $file)
  dfile=$(echo $file | sed s@/opengameart/files@/opengameart/unpacked@)
  path=$dfile
  count=`s3cmd ls "$path" | wc -l`
  if [[ $count -gt 0 ]]; then
    echo "Skipping $file"
    sleep 0.1
    continue
  fi
  rm -rf extract/unpacked
  mkdir extract/unpacked &&
  s3cmd get "$file" "extract/$bname" &&
  unar "extract/$bname" -o extract/unpacked &&
  (
    cd extract/unpacked
    find -type f | while read efile; do
      cfile=$(echo "$efile" | cut -b3-)
      s3cmd put "$efile" "$dfile/$cfile" --acl-public
      sleep 0.1
      #s3cmd setacl $dfile/$cfile --acl-public
    done
  )
  rm -f "extract/$bname"
  sleep 2
done
