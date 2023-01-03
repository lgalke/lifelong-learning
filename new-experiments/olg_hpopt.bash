DATA="dblp-easy"
YEAR=2004
ANNUAL_EPOCHS=200
NLAYERS=1
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS"
OLG_ARGS="--open_learning doc --doc_threshold 0.5 --doc_class_weights"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $ANNUAL_EPOCHS"
OUTFILE="results/revision-results/tg-gat-hopt.csv"

# Exit on error
set -e

for SEED in 101 102 103; do
  for LR in "0.1" "0.05" "0.01" "0.005" "0.001" "0.0005"; do
    for HISTORY in 1 3 6 25; do
      # MLP
      # python3 run_experiment_new.py --seed "$SEED" --model mlp --n_hidden 64 --start warm --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE --backend dgl"
      # python3 run_experiment_new.py --seed "$SEED" --model mlp --n_hidden 64 --start cold --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE --backend dgl"

      # GAT
      # python3 run_experiment_new.py --seed "$SEED" --model gat --n_hidden 64 --start warm --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE --backend dgl"
      # python3 run_experiment_new.py --seed "$SEED" --model gat --n_hidden 64 --start cold --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE --backend dgl"

      # GraphSAGE (used for Exp 4)
      # python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start warm --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE --backend dgl"
      # python3 run_experiment_new.py --seed "$SEED" --model gs-mean --n_hidden 32 --start cold --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE --backend dgl"


      # SGC (2022-12-11)
      # python3 run_experiment_new.py --seed "$SEED" --model sgnet --n_hidden 32 --start warm --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE" --backend geometric
      # python3 run_experiment_new.py --seed "$SEED" --model sgnet --n_hidden 32 --start cold --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE" --backend geometric

      # GAT (2022-12-11)
      python3 run_experiment_new.py --seed "$SEED" --model gat --n_hidden 32 --start warm --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE" --backend dgl
      python3 run_experiment_new.py --seed "$SEED" --model gat --n_hidden 32 --start cold --lr $LR --history $HISTORY $ARGS $PRETRAIN_ARGS --dataset "$DATA" $OLG_ARGS --save "$OUTFILE" --backend dgl
    done
  done
done
