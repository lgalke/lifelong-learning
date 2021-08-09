DATA="dblp-hard"
YEAR=2004
ANNUAL_EPOCHS=200
NLAYERS=1
BACKEND="dgl"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS --backend $BACKEND"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $ANNUAL_EPOCHS"
OUTFILE="results/open_DOC_CW_risk-reduction_alpha-values.csv"

# Exit on error
set -e

HPARAMS=(
	# "--history 1 --start cold --lr 0.005"
	# "--history 1 --start warm --lr 0.0005"
	# "--history 3 --start cold --lr 0.005"
	"--history 3 --start warm --lr 0.001"
	# "--history 6 --start cold --lr 0.01"
	# "--history 6 --start warm --lr 0.005"
	# "--history 25 --start cold --lr 0.01"
	# "--history 25 --start warm --lr 0.005"
)


for SEED in 10 11 12 13 14; do
	# for DOC_THRESHOLD in "0.5" "0.25" "0.0"; do
	for DOC_THRESHOLD in "1.00"; do
		for DOC_ALPHA in "0.5" "1.0" "1.5" "2" "2.5" "3.0"; do
			OLG_ARGS="--open_learning doc --doc_threshold $DOC_THRESHOLD --doc_reduce_risk --doc_alpha $DOC_ALPHA --doc_class_weights"
			for i in ${!HPARAMS[@]}; do
				echo "${HPARAMS[$i]}"
				python3 run_experiment_new.py ${HPARAMS[$i]} --seed "$SEED" --model gs-mean --n_hidden 32 $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE"
			done
		done

	done
done
