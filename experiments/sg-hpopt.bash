DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
OUTFILE="results/sg-hpopt.csv"
ARGS="--weight_decay 0 --rescale_lr 1. --rescale_wd 1."
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"

set -e

for LR in "0.1" "0.05" "0.01" "0.005" "0.001" "0.0005"; do
	for START in "cold" "warm"; do
		for HISTORY in 1 3 6 25; do
			for SEED in 101 102 103; do
				python3 run_experiment.py --history $HISTORY --seed "$SEED" --model sgnet --start $START --lr $LR --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
			done
		done
	done
done

