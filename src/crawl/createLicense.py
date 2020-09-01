#!/usr/bin/env python3

import sys
import os
from glob import glob
from bs4 import BeautifulSoup

blacklist = open('blacklist.txt', 'w')
files = sorted(glob('content/*.html'))
print('<html><head><meta charset="utf-8"></head><body>')
print('<table>')
for i, fname in enumerate(files):
    #print(fname)
    #if not 'nebula' in fname:
    #    continue
    fd = open(fname)
    html_doc = fd.read()
    fd.close()
    soup = BeautifulSoup(html_doc, 'html.parser')
    for span in soup.find_all('span'):
        cls = span.get('class')
        if cls is not None:
            if 'username' in cls:
                username = span.get_text()
    attribution = []
    license = []
    for div in soup.find_all('div'):
        cls = div.get('class')
        if cls is not None:
            if 'license-name' in cls:
                license.append(div.get_text())
            if 'field-name-field-copyright-notice' in cls:
                attribution.append(div.get_text())
            if 'field-name-field-art-attribution' in cls:
                attribution.append(div.get_text())
            if 'field' in cls:
                dtl = div.get_text().lower()
                if 'attribution' in dtl:
                    index = dtl.index('attribution')
                    a = div.get_text()[index:]
                    #print(a, file=sys.stderr)
                    attribution.append(a)
    license = ', '.join(license)
    if license == 'GPL 2.0':
        blacklist.write(fname + '\n')
    page = 'http://opengameart.org/' + fname.replace('.html', '')
    if not attribution:
        attribution = 'None'
    else:
        attribution = '<br/>'.join(attribution)

    #print("username", username, page, license, attribution)
    print('<tr>')

    print('<td>')
    print('<h1>Username: ' + username + '</h1>')
    print('</td>')

    print('<td>')
    print('<p>')
    print('<a href="' + page + '">' + page + '</a>')
    print('</p>')
    print('</td>')

    print('<td>')
    print('<p>')
    print('<h2>License:' + license + '</h2>')
    print('</p>')
    print('</td>')

    print('<td>')
    print('<h2>Attribution:</h2>')
    print('<p>')
    print(attribution)
    print('</p>')
    print('<br/>')
    print('</td>')

    print('</tr>')
    print(i, len(files), fname, file=sys.stderr)
print('</table>')
print('</body>')
print('</html>')
blacklist.close()
