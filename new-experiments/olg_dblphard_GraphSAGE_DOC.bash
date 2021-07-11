DATA="dblp-hard"
YEAR=2004
ANNUAL_EPOCHS=200
NLAYERS=1
BACKEND="dgl"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS --backend $BACKEND"
# OLG_ARGS="--open_learning doc --doc_threshold 0.5 --doc_reduce_risk --doc_alpha 3.0"
OLG_ARGS="--open_learning doc --doc_threshold 0.5"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $ANNUAL_EPOCHS"
OUTFILE="results/olg_dblphard_GraphSAGE-first-exp-DOC.csv"

# Exit on error
set -e

HPARAMS=(
	"--history 1 --start cold --lr 0.005"
	"--history 1 --start warm --lr 0.001"
	"--history 3 --start cold --lr 0.005"
	"--history 3 --start warm --lr 0.005"
	"--history 6 --start cold --lr 0.01"
	"--history 6 --start warm --lr 0.005"
	"--history 25 --start cold --lr 0.01"
	"--history 25 --start warm --lr 0.01"
)

for SEED in 10; do
	for i in ${!HPARAMS[@]}; do
		echo "${HPARAMS[$i]}"
		python3 run_experiment_new.py ${HPARAMS[$i]} --seed "$SEED" --model gs-mean --n_hidden 32 $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE"
	done
done
