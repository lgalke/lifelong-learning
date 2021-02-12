DATA="dblp-hard"
YEAR=2004
HISTORY=3
NLAYERS=1 # 2 graph conv layers
ARGS="--start warm --n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --history $HISTORY --rescale_lr 1. --rescale_wd 1."
DATA_ARGS="--data "$DATA" --t_start $YEAR"
STATIC_MODEL_ARGS="--initial_epochs 400 --annual_epochs 0"
UPTRAIN_MODEL_ARGS="--initial_epochs 0 --annual_epochs 200"
OUTFILE="results/ablation-hard.csv"

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --lr "0.001" $STATIC_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --lr "0.001" $UPTRAIN_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gs-mean --n_hidden 32 --lr "0.001" $STATIC_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gs-mean --n_hidden 32 --lr "0.005" $UPTRAIN_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --lr "0.005" $STATIC_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --lr "0.01" $UPTRAIN_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
done
