#!/bin/bash
 
python -m earlyexitnet.cli -m b_lenet_se -np ./test.txt -mp ./trained_models/b_lenet_se.pth -rn "run notes example" -t1 0.99 -entr 0.01 -gpu 0 -nw 8 -bste 100 -cf 1 2 3 4
