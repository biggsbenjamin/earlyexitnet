#!/bin/bash
 
python -m earlyexitnet.cli -m b_lenet_cifar -d cifar10 -bbe 100 -jte 50 -rn "run notes example" -t1 0.75 -entr 0.01 -gpu 0 -nw 8 -cf 0 1
