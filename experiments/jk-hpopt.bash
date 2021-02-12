# origin: ../experiments-journal/jk-hpopt3layers.bash but changed LAYERS=2
DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2
OUTFILE="results/jk-hpopt.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5  --rescale_lr 1. --rescale_wd 1."
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
HIDDEN=16

set -e

for LR in "0.1" "0.05" "0.01" "0.005" "0.001" "0.0005"; do
	for START in "cold" "warm"; do
		for HISTORY in 1 3 6 25; do
			for SEED in 101 102 103; do
				python3 run_experiment_new.py --history $HISTORY --seed "$SEED" --backend dgl --model jknet-sageconv --n_hidden $HIDDEN --start $START --lr $LR --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
			done
		done
	done
done

