#!/bin/bash
 

thresh_step="0.5"
thresh_min="0.0"
thresh_max="0.99"

OUTPUT_FILE_PATH="./threshold_range.txt"
COMPARE_FUNCS="1"
BATCH_SIZE="500"

python -m earlyexitnet.cli -m b_lenet_se -np $OUTPUT_FILE_PATH -mp ./trained_models/b_lenet_se.pth -rn "run notes example" -gpu 0 -nw 8 -bste $BATCH_SIZE -cf $COMPARE_FUNCS -tr $thresh_min $thresh_max -ts $thresh_step
