#!/bin/bash

# Training resnet8 (not necessarily well) but very quickly on a chonky gpu

python -m earlyexitnet.cli -m resnet8_2ee -d cifar10 -bstr 32 -bbe 0 -jto 'adam-wd-cos-sched' -vf 11  -t1 0.6 -entr 0.02 -cf 0 1 2 3 4 -bste 1000 -rn "trying higher starting lr and longer t0" -jte 200 -nw 16 -mp "./trained_models/resnet8_bb_t1acc88.pth" -gpu 2
