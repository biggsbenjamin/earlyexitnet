#!/bin/bash
 
python -m earlyexitnet.cli -m b_lenet -mp ../outputs/pre_Trn_bb_2023-07-06_234313/pretrn-joint-e30-2023-07-06_235234.pth -rn "run notes example" -t1 0.75 -entr 0.01 -gpu 0