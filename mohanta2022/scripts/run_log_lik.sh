#!/bin/bash

export FLYMAZERL_PATH=/groups/turner/turnerlab/Rishika/FlYMazeRL

# bsub -n1 -J "m0" -o "m0.output" "python log_lik.py 0"
# sleep .5
# bsub -n1 -J "m1" -o "m1.output" "python log_lik.py 1"
# sleep .5
bsub -n1 -J "m2" -o "m2.output" "python log_lik.py 2"
sleep .5
bsub -n1 -J "m3" -o "m3.output" "python log_lik.py 3"
sleep .5
bsub -n1 -J "m4" -o "m4.output" "python log_lik.py 4"
sleep .5
bsub -n1 -J "m5" -o "m5.output" "python log_lik.py 5"
sleep .5
bsub -n1 -J "m6" -o "m6.output" "python log_lik.py 6"
sleep .5
bsub -n1 -J "m7" -o "m7.output" "python log_lik.py 7"
sleep .5
bsub -n1 -J "m8" -o "m8.output" "python log_lik.py 8"
sleep .5
bsub -n1 -J "m9" -o "m9.output" "python log_lik.py 9"
sleep .5
bsub -n1 -J "m10" -o "m10.output" "python log_lik.py 10"
sleep .5
bsub -n1 -J "m11" -o "m11.output" "python log_lik.py 11"
sleep .5
bsub -n1 -J "m12" -o "m12.output" "python log_lik.py 12"
sleep .5
bsub -n1 -J "m13" -o "m13.output" "python log_lik.py 13"
sleep .5
bsub -n1 -J "m14" -o "m14.output" "python log_lik.py 14"
sleep .5
bsub -n1 -J "m15" -o "m15.output" "python log_lik.py 15"
sleep .5
bsub -n1 -J "m16" -o "m16.output" "python log_lik.py 16"
sleep .5
bsub -n1 -J "m17" -o "m17.output" "python log_lik.py 17"
sleep .5
bsub -n1 -J "m18" -o "m18.output" "python log_lik.py 18"
sleep .5
bsub -n1 -J "m19" -o "m19.output" "python log_lik.py 19"
sleep .5
bsub -n1 -J "m20" -o "m20.output" "python log_lik.py 20"
sleep .5
bsub -n1 -J "m21" -o "m21.output" "python log_lik.py 21"
sleep .5
bsub -n1 -J "m22" -o "m22.output" "python log_lik.py 22"
sleep .5
bsub -n1 -J "m23" -o "m23.output" "python log_lik.py 23"
sleep .5
bsub -n1 -J "m24" -o "m24.output" "python log_lik.py 24"
sleep .5