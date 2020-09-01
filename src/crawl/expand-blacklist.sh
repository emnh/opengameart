for file in $(cat blacklist.txt); do
  grep -o "https://opengameart.org/sites/default/files/[^\"']*" $file
done | fgrep -v /files/styles | sort -u | sed 's@.*/sites/default/@@'
