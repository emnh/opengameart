#!/bin/bash
for page in `seq 0 86`; do
  #wget -c 'https://opengameart.org/art-search-advanced?keys=&title=&field_art_tags_tid_op=or&field_art_tags_tid=&name=&field_art_type_tid[0]=9&field_art_licenses_tid[17981]=17981&field_art_licenses_tid[2]=2&field_art_licenses_tid[17982]=17982&field_art_licenses_tid[3]=3&field_art_licenses_tid[6]=6&field_art_licenses_tid[5]=5&field_art_licenses_tid[10310]=10310&field_art_licenses_tid[4]=4&field_art_licenses_tid[8]=8&field_art_licenses_tid[7]=7&sort_by=count&sort_order=DESC&items_per_page=144&collection=&page='$page'&Collection=' -O page$page.html
  wget -c 'https://opengameart.org/art-search-advanced?keys=&title=&field_art_tags_tid_op=or&field_art_tags_tid=&name=&field_art_type_tid%5B0%5D=9&field_art_licenses_tid%5B0%5D=17981&field_art_licenses_tid%5B1%5D=2&field_art_licenses_tid%5B2%5D=17982&field_art_licenses_tid%5B3%5D=3&field_art_licenses_tid%5B4%5D=6&field_art_licenses_tid%5B5%5D=5&field_art_licenses_tid%5B6%5D=10310&field_art_licenses_tid%5B7%5D=4&field_art_licenses_tid%5B8%5D=8&field_art_licenses_tid%5B9%5D=7&sort_by=score&sort_order=DESC&items_per_page=144&Collection=&page='$page -O pages2/page$page.html
  sleep 5
done
