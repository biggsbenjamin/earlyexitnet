#!/bin/bash
 

thresh_step="0.05"
thresh_min="0.0"
thresh_max="0.99"

OUTPUT_FILE_PATH="./threshold_range.txt"
COMPARE_FUNCS="1 2 4"
BATCH_SIZE="500"

num_tests=$(bc<<<"($thresh_max - $thresh_min)/$thresh_step")

for (( i=0; $i<=$num_tests; i++ )); do
      thresh_val=$(bc<<<"$thresh_min + $thresh_step * $i")
      msg=$(printf "Test %s/%s, threshold:%s" "$i" "$num_tests" "$thresh_val")

      set -o xtrace
      python -m earlyexitnet.cli -m b_lenet_se -np $OUTPUT_FILE_PATH -mp ./trained_models/b_lenet_se.pth -rn "$msg" -t1 $thresh_val -entr 0.01 -gpu 0 -nw 8 -bste $BATCH_SIZE  -cf $COMPARE_FUNCS

      set +o xtrace
done

