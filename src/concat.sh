#!/bin/bash
cd predict &&
find . -maxdepth 1 -type f -print0 -name \*.np |
  sort -z |
  xargs -0 cat -- >>../predict.out
