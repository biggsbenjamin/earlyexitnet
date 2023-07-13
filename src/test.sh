#!/bin/bash
 
python -m earlyexitnet.cli -m b_lenet_se -mp ./trained_models/b_lenet_se.pth -rn "run notes example" -t1 0.75 -entr 0.01 -gpu 0 -nw 8 -bste 2000
