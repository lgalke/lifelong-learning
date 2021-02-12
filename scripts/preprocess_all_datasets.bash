set -e 

# Preprocess datasets 'dblp-easy', 'dblp-hard', and 'pharmabio' assuming they reside within a ./data/<dataset> directory relative to script execution cwd.
# Important: t_zero is usually set to one task before the first evaluation task (such as t_start - 1)
# Output will be saved to ./data/<dataset>/<prepocessed_dataset>

#################
### DBLP-EASY ###
#################
DATASET="data/dblp-easy"
TZERO="2003"
echo "Preprocessing $DATASET"
for BACKEND in "geometric" "dgl"; do
    for HISTORY in "1" "3" "6" "25"; do
        python3 preprocess_dataset.py $DATASET --t_zero $TZERO --backend $BACKEND --history $HISTORY
    done
done


#################
### DBLP-HARD ###
#################
DATASET="data/dblp-hard"
TZERO="2003"
echo "Preprocessing $DATASET"
for BACKEND in "geometric" "dgl"; do
    for HISTORY in "1" "3" "6" "25"; do
        python3 preprocess_dataset.py $DATASET --t_zero $TZERO --backend $BACKEND --history $HISTORY
    done
done


#################
### PHARMABIO ###
#################
DATASET="data/pharmabio"
TZERO="1998"
echo "Preprocessing $DATASET"
for BACKEND in "geometric" "dgl"; do
    for HISTORY in "1" "4" "8" "21"; do
        python3 preprocess_dataset.py $DATASET --t_zero $TZERO --backend $BACKEND --history $HISTORY
    done
done


