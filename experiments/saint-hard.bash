DATA="dblp-hard"
YEAR=2004
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2
OUTFILE="results/saint-hard.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5  --rescale_lr 1. --rescale_wd 1. --n_hidden 16 --sampling rw --batch_size 0.5 --saint_coverage 0"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
hparams=("--history 1 --start cold --lr 0.005"
            "--history 3 --start cold --lr 0.005"
            "--history 6 --start cold --lr 0.005"
            "--history 25 --start cold  --lr 0.01"
            "--history 1 --start warm --lr 0.001"
            "--history 3 --start warm --lr 0.001"
            "--history 6 --start warm --lr 0.001"
            "--history 25 --start warm --lr 0.005")

set -e
for SEED in 201 202 203 204 205 206 207 208 209 210; do
  for i in ${!hparams[@]}; do
    echo ${hparams[$i]}
    python3 run_experiment_new.py ${hparams[$i]} --inductive --seed "$SEED" --backend geometric --model graphsaint --variant jknet-graphconv --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  done
done
 
