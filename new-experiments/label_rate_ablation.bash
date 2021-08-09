YEAR=2004
INITIAL_EPOCHS=0
ANNUAL_EPOCHS=200
NLAYERS=1
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS --backend dgl"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
OUTFILE="results/label_rate_ablation_final.csv"


set -e


for SEED in 10 11 12 13 14; do
  for LABELRATE in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"; do
    HISTORY=1
    DATA="/media/nvme1n1/lgalke/datasets/Incremental-GNNs/label_rate_ablation/dblp-hard-tzero2003-history$HISTORY-dgl-$LABELRATE"
    # HISTORY 1
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start cold --lr "0.001" --history 1 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start warm --lr "0.001" --history 1 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start cold --lr "0.001" --history 1 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start warm --lr "0.001" --history 1 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004
    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start cold --lr "0.005" --history 1 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start warm --lr "0.001" --history 1 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # # HISTORY 3
    HISTORY=3
    DATA="/media/nvme1n1/lgalke/datasets/Incremental-GNNs/label_rate_ablation/dblp-hard-tzero2003-history$HISTORY-dgl-$LABELRATE"
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start cold --lr "0.001" --history 3 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start warm --lr "0.001" --history 3 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start cold --lr "0.005" --history 3 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start warm --lr "0.005" --history 3 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004
    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start cold --lr "0.005" --history 3 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start warm --lr "0.005" --history 3 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # HISTORY 6
    HISTORY=6
    DATA="/media/nvme1n1/lgalke/datasets/Incremental-GNNs/label_rate_ablation/dblp-hard-tzero2003-history$HISTORY-dgl-$LABELRATE"
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start cold --lr "0.005" --history 6 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start warm --lr "0.001" --history 6 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start cold --lr "0.01" --history 6 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start warm --lr "0.005" --history 6 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004
    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start cold --lr "0.05" --history 6 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start warm --lr "0.01" --history 6 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # HISTORY 25 
    HISTORY=25
    DATA="/media/nvme1n1/lgalke/datasets/Incremental-GNNs/label_rate_ablation/dblp-hard-tzero2003-history$HISTORY-dgl-$LABELRATE"
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start cold --lr "0.005" --history 25 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start warm --lr "0.001" --history 25 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"

    # Next two lines may fail on 12GB GPUs, maybe need to run on CPU instead
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start cold --lr "0.01" --history 25 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004
    python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start warm --lr "0.005" --history 25 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE" --label_rate $LABELRATE --t_start 2004

    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start cold --lr "0.05" --history 25 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
    # python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start warm --lr "0.05" --history 25 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
  done
done
