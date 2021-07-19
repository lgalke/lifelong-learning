DATA="dblp-hard"
YEAR=2004
ANNUAL_EPOCHS=200
NLAYERS=1
BACKEND="dgl"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS --backend $BACKEND"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $ANNUAL_EPOCHS"
OUTFILE="results/open_DOC_dummy.csv"

# Exit on error
set -e

HPARAMS=(
	# "--history 1 --start cold --lr 0.005"
	# "--history 1 --start warm --lr 0.001"
	# "--history 3 --start cold --lr 0.005"
	# "--history 3 --start warm --lr 0.005"
	# "--history 6 --start cold --lr 0.01"
	"--history 6 --start warm --lr 0.005"
	# "--history 25 --start cold --lr 0.01"
	# "--history 25 --start warm --lr 0.01"
)

for SEED in 10; do
	for DOC_THRESHOLD in "0.5"; do
		OLG_ARGS="--open_learning doc --doc_threshold $DOC_THRESHOLD"
		for i in ${!HPARAMS[@]}; do
			echo "${HPARAMS[$i]}"
			python3 run_experiment_new.py ${HPARAMS[$i]} --seed "$SEED" --model gs-mean --n_hidden 32 $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE" --save_logits_dir "results/doc_h6_warm_logits_targets"
		done
	done
done
