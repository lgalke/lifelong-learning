# Lifelong Learning of Graph Neural Networks for Open-World Node Classification

## Papers

Lukas Galke, Iacopo Vagliano, Benedikt Franke, Tobias Zielke, Marcel Hoffmann, Ansgar Scherp (2023). [Lifelong Learning on Evolving Graphs Under the Constraints of Imbalanced Classes and New Classes](https://authors.elsevier.com/a/1h1SX3BBjKnulZ). *Neural Networks* 164, 156-176.

Lukas Galke, Benedikt Franke, Tobias Zielke, Ansgar Scherp (2021). [Lifelong Learning of Graph Neural Networks for Open-World Node Classification](https://doi.org/10.1109/IJCNN52387.2021.9533412), In *2021 International Joint Conference on Neural Networks (IJCNN)*, IEEE.

```
@article{galke2023lifelong,
  title = {Lifelong learning on evolving graphs under the constraints of imbalanced classes and new classes},
  journal = {Neural Networks},
  volume = {164},
  pages = {156-176},
  year = {2023},
  issn = {0893-6080},
  doi = {https://doi.org/10.1016/j.neunet.2023.04.022},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608023002083},
  author = {Lukas Galke and Iacopo Vagliano and Benedikt Franke and Tobias Zielke and Marcel Hoffmann and Ansgar Scherp}
}

@inproceedings{galke2021lifelong,
  author={Galke, Lukas and Franke, Benedikt and Zielke, Tobias and Scherp, Ansgar},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  title={Lifelong Learning of Graph Neural Networks for Open-World Node Classification},
  year={2021},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN52387.2021.9533412}
}
```

## Installation

1. Setup a python virtual environment (recommended)
2. Install [pytorch](https://pytorch.org/get-started/locally/) as suited to your
   OS / python package manager
3. Install [dgl](https://www.dgl.ai/pages/start.html) as suited to your
   OS / python package manager / CUDA version
4. Install [torch-geometric](https://github.com/rusty1s/pytorch_geometric)
5. Install other requirements via `pip install -r requirements.txt` within your
   copy of this repository. This will include mainly `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

In the `requirements.txt` file  we list the versions we have used for our experiments. Newer versions of standard packages will likely work fine.


## Get the datasets

The three datasets of our paper are available [on zenodo](https://zenodo.org/record/3764770).
Download the zip files and extract them into the `data` subdirectory, such that the structure looks exactly like this:

- `data/dblp-easy`
- `data/dblp-hard`
- `data/pharmabio`

## Example call to run an experiment

The following exemplary command will run an experiment with a GraphSAGE model with two graph convolutoinal layers and a hidden dimension of 32 on the `dblp-easy` dataset starting evaluation at task corresponding to year 2004 while using 200 annual epochs.

```
python3 run_experiment.py --seed 42 --backend dgl --model gs-mean --n_hidden 32 --start warm --lr "0.005" --history 3 --n_layers 1 --weight_decay 0 --dropout 0.5 --initial_epochs 0 --annual_epochs 200 --dataset "dblp-easy" --t_start 2004 --save "results.csv"                       
```

The results.csv file can be reused for multiple runs (e.g. with different seeds, different models, different datasets), the script appends new results to the file.
Consult `python3 run_experiment.py -h` for more information. The other variant of our run script `run_experiment_new.py` is similar, but operates on preprocessed tasks (see below).


## Tabularize results

To bring the results into a human readable form, we provide a `tabularize.py` script.
This script takes one or more results files as input and groups them depending on the `-g` parameter, which can be multiple attributes.
Example: *Aggregate results by dataset, history size, model, restart configuration, and learning rate*.

```
python3 tabularize.py results.csv -g dataset history model start initial_lr
```

The script `tabularize_forward_transfer.py` uses the exact grouping as we use for the results of the paper and also computes Forward Transfer (averaged difference between warm and cold restarts).
Note that each configuration must have a value for both `start=warm` and `start=cold`, such that Forward Transfer can be computed.


## Visualize results

You can visualize with the `visualize.py` script:

```
python3 visualize.py --style "window size %RF" --hue model --col dataset --row start --save plot.png results.csv
```

where results.csv is the file where you've aggregated the results. You can also provide multiple results files, then they will be concatenated before plotting.

## Preprocessing tasks for multiple experiments

Constructing tasks dynamically may take some time during experiments.
The tasks can be preprocessed with the script `preprocess_datasets.py`
**Important:** the Argument t_zero must be one task **before** the first evaluation task, i.e., if you want to start evaluation at time 2004, you must preprocess tasks starting at time 2003.

Example to preprocess tasks for dblp-easy starting at year 2003 with a history size of 3 for models implemented via `dgl`:

```
python3 preprocess_dataset.py data/dblp-easy` --t_zero 2003 --history 3 --backend dgl data/dblp-easy
```

In `scripts/preprocess_all_datasets.bash`, you find a shorthand to preprocess all datasets in all history size configurations from the paper for both backends.
This shorthand should be started with the repository's root as working directory and expects the directory structure of `data/` as described above.

Then, you need to use the version of the run script that uses preprocessed tasks, namely `run_experiment_new.py`.
The interface of the script is nearly the same as the one of `run_experiment.py`. 
One difference is that reproducing our ablation study for comparison with once-trained static models is only possible with `run_experiment.py`.

## Full reproduction of the paper's experiments

In the `experiments/` directory, you find bash scripts to re-run all our experiments.

## Issues?

**Note:** Such that the python/bash scripts within this repository work properly, you must call them from the root directory of this repository.

If there are any problems with using this repository, feel free to file an issue.

## File Descriptions

| File                              | Description                                                                 |
| -                                 | -                                                                           |
| analysis                          | scripts to perform analyses                                                 |
| datasets.py                       | dataset loading                                                             |
| lifelong_learning.py              | lifelong learning module                                                    |
| drift_magnitude.py                | drift magnitude module                                                      |
| preprocess_dataset.py             | Create a Lifelong Learning Graph Dataset by preprocessing tasks             |
| experiments                       | Bash scripts to reproduce experiments                                       |
| scripts                           | Other bash scripts to preprocess/compute tdiff distribution                 |
| models                            | GNN implementations                                                         |
| README.md                         | this file                                                                   |
| requirements.txt                  | dependencies                                                                |
| run_experiment.py                 | main entry point for running a single experiment                            |
| run_experiment_new.py             | main entry point for running a single experiment **with preprocessed data** |
| tabularize.py                     | aggregate results into table                                                |
| tabularize_forward_transfer.py    | aggregate results into table including Forward Transfer computation         |
| visualize.py                      | visualize results                                                           |


## Notes

- The experiments for inductive vs transductive learning can be found in a different [repository](https://github.com/lgalke/gnn-pretraining-evaluation).
