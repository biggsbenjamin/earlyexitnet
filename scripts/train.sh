#!/bin/bash

# Example pf training a 2 exit model

python -m earlyexitnet.cli -m b_lenet_cifar -d cifar10 -bbe 300 -jte 150 -t1 0.75 -entr 0.01 -gpu 0 -nw 8 -cf 0 1
