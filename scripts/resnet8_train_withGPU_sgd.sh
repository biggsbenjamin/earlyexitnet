#!/bin/bash
 
# Training resnet8 (not necessarily well) but very quickly on a chonky gpu

python -m earlyexitnet.cli -m resnet8 -d cifar10 -bstr 128 -bbe 300 -bbo 'sgd-plat-sched' -vf 5 -gpu 0 -nw 64 -bste 1000 -rn "For fun and profit, let's go! maybe actually useful"
