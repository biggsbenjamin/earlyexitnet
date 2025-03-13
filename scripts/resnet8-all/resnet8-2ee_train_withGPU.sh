#!/bin/bash

# Training resnet8_2EE (not necessarily well) but very quickly on a chonky gpu
# using pre-trained starting point

python -m earlyexitnet.cli -m resnet8_2ee -d cifar10 -bbe 0 -jte 200 -t1 0.6 -entr 0.02  -nw 16 -cf 0 1 -vf 10 -bstr 32 -jto 'adam-wd' -bste 1000 -rn "repeating a bit of what ive done" -mp "./trained_models/resnet8_bb_t1acc88.pth" -gpu 2
