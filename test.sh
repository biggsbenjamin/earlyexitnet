#!/bin/bash
 
# python -m earlyexitnet.cli -m b_lenet_cifar -d cifar10 -mp ./trained_models/b_lenet_cifar10.pth -t1 0.75 -entr 0.01 -gpu 0 -nw 8 -bste 500 -cf 0 1 2 3 4 -sr True
# python -m earlyexitnet.cli -m b_lenet_se -mp ./trained_models/b_lenet_se.pth -t1 0.75 -entr 0.01 -gpu 0 -nw 8 -bste 500 -cf 0 1 2 3 4 -sr True

python -m earlyexitnet.cli --dataset cifar10 -m resnet8_2ee -mp ./trained_models/resnet8_2ee_230721_jnt-frm-scrtch.pth -t1 0.8 -entr 0.65 -gpu 0 -nw 8 -bste 500 -cf 0 1 2 3 4 -sr True
