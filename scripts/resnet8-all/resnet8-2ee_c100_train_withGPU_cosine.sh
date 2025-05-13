#!/bin/bash

# Training resnet8 (not necessarily well) but very quickly on a chonky gpu

python -m earlyexitnet.cli -m resnet8_2ee -d cifar100 -bstr 64 -bbe 0 -jto 'adam-wd-cos-sched' -vf 11 -t1 0.5 -entr 0.03 -cf 0 1 2 3 4  -nw 8 -bste 1000 -rn "cifar100 dataset, 2 exit version" -jte 500 -mp "./trained_models/resnet8_bb_cifar100_t160_250313.pth" -gpu 3
