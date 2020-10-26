#!/bin/bash

i=5
number=100

while [ $i -le $number ]
do
    j=`echo $i / 100.0 |bc -l`
    echo $j
    python3 object_detection_tf.py --file ./sample/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00023006.mp4 --threshold_score $j
    i=$(( $i + 5 ))
done