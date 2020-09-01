echo {
for cfile in content/*; do
  fgrep 'File(s)' $cfile |
  grep -o 'https://opengameart.org/sites/default/files/[^"]*' |
  while read file; do
    echo "\"$file\"": "\"$cfile\"",
  done
done
echo }
