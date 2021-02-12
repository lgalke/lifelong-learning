DATA="pharmabio"
YEAR=1999
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2
OUTFILE="results/jk-pharma.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5  --rescale_lr 1. --rescale_wd 1. --n_hidden 16"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"

hparams=("--history 1 --start cold --lr 0.005"
            "--history 4 --start cold --lr 0.005"
            "--history 8 --start cold --lr 0.005"
            "--history 21 --start cold --lr 0.01"
            "--history 1 --start warm --lr 0.001"
            "--history 4 --start warm --lr 0.001"
            "--history 8 --start warm --lr 0.001"
            "--history 21 --start warm --lr 0.005")

set -e

for SEED in 201 202 203 204 205 206 207 208 209 210; do
  for i in ${!hparams[@]}; do
    echo ${hparams[$i]}
    python3 run_experiment_new.py ${hparams[$i]} --seed "$SEED" --backend dgl --model jknet-sageconv --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  done
done
 
