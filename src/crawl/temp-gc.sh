#!/bin/bash
grep -o 'href="/content/[^"]*"' pages2/*.html |
cut -f2 -d= |
sed 's/"//g' |
sort -u | while read urlpart; do
  bname=$(basename $urlpart)
  echo wget -c 'https://opengameart.org/'$urlpart -O content2/$bname.html
  #sleep 2
done
