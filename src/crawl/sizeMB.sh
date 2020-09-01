#!/bin/bash
num=$(grep -o ' length=[0-9]*' content/* |
sed s/length=// |
cut -f2 -d: |
awk '{s+=$1} END {print s}')
expr $num / 1024 / 1024
