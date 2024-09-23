#!/usr/bin/bash

for file in $(ls -1 dot*.png); do convert $file -gravity north -extent 800x500 $file; done
convert -delay 50 -loop 0 `ls -1 dot*.png | sort -n -k1.4` gen_graph.gif
