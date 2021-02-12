DATA="dblp-easy"
YEAR=2004
HISTORY=3
NLAYERS=1 # 2 graph conv layers
ARGS="--start warm --n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --history $HISTORY --rescale_wd 1. --rescale_lr 1."
DATA_ARGS="--data "$DATA" --t_start $YEAR"
STATIC_MODEL_ARGS="--initial_epochs 400 --annual_epochs 0"
UPTRAIN_MODEL_ARGS="--initial_epochs 0 --annual_epochs 200"
OUTFILE="results/ablation-hpopt.csv"

for SEED in 101 102 103; do
  for LR in "0.0005" "0.001" "0.005" "0.01" "0.05" "0.1"; do
    # STATIC
    python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --lr $LR $STATIC_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
    python3 run_experiment.py --seed "$SEED" --model gs-mean --n_hidden 32 --lr $LR $STATIC_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
    python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --lr $LR $STATIC_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"

    # UPTRAIN
    python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --lr $LR $UPTRAIN_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
    python3 run_experiment.py --seed "$SEED" --model gs-mean --n_hidden 32 --lr $LR $UPTRAIN_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
    python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --lr $LR $UPTRAIN_MODEL_ARGS $ARGS $DATA_ARGS --save "$OUTFILE"
  done
done
