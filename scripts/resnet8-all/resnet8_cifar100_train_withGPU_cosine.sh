#!/bin/bash

# Training resnet8 (not necessarily well) but very quickly on a chonky gpu

python -m earlyexitnet.cli -m resnet8_bb -d cifar100 -bstr 32 -bbe 2000 -bbo 'adam-wd-cos-sched' -vf 11 -gpu 1 -nw 64 -bste 1000 -rn "cifar100 dataset, aiming for ~70 t1"
