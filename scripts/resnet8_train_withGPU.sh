#!/bin/bash
 
# Training resnet8 (not necessarily well) but very quickly on a chonky gpu

python -m earlyexitnet.cli -m resnet8 -d cifar10 -bstr 500 -bbe 300 -vf 10 -gpu 1 -nw 64 -bste 10000 -rn "For fun and profit, let's go!"
