import argparse
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("results_csv_file", help="Path to results file", nargs='+')
parser.add_argument("--latex", default=False, action='store_true', help="Produce latex output")
parser.add_argument("--save", help="Path to save resulting table")

args = parser.parse_args()
print("Loading data:", args.results_csv_file)
df = pd.read_csv(args.results_csv_file[0])
print("N =",len(df))
for path in args.results_csv_file[1:]:
    print("Adding data:", path)
    add_data = pd.read_csv(path)
    df = pd.concat([df, add_data], axis=0, ignore_index=True)
    print("N =",len(df))

def SD(values):
    """ Just to ensure we use ddof=1 """
    return values.std(ddof=1)

def SE(values):
    return values.std(ddof=1) / np.sqrt(values.count())

def forward_transfer(df):
    print("Computing forward transfer")
    first_task_per_dataset = df.groupby('dataset', as_index=False)['year'].min()
    print("Dropping first task per dataset:", first_task_per_dataset, sep='\n')
    for dataset, first_task in first_task_per_dataset.itertuples(index=False):
        idx = df[(df['dataset'] == dataset) & (df['year'] == first_task)].index
        df = df.drop(idx, axis=0)

    df = df.pivot_table(index=['dataset', 'model', 'history', 'year'], columns=['start'], aggfunc='mean')
    df['FWT'] = df['accuracy']['warm'] - df['accuracy']['cold']
    df.drop('accuracy', axis=1, inplace=True)
    # Average out the tasks (years)
    fwt = df.groupby(['dataset','model','history']).FWT.mean()
    return fwt

# dataset,seed,model,variant,n_params,n_hidden,n_layers,dropout,history,sampling,batch_size,saint_coverage,limited_pretraining,initial_epochs,initial_lr,initial_wd,annual_epochs,annual_lr,annual_wd,
# start,decay,year,epoch,f1_macro,accuracy

for col in ['annual_lr', 'sampling', 'batch_size', 'variant']:
    if col in df and len(df[col].unique()) > 1:
        print(f"[warn] Don't apply this to hyperparameter optimization results! Will not group by column '{col}'")

# TODO include 'seed'?
groupby_cols = ['dataset','model','history']

# Selet subset of interesting columns
df = df[(groupby_cols + ['start', 'accuracy', 'year'])]
# print(df.head())

# print("N =", len(df))
# print("Grouping by:", groupby_cols)
# grouped_df = df.groupby(groupby_cols, as_index=False).accuracy.mean()
# print(grouped_df)


fwt = forward_transfer(df)

# df = df.pivot_table(index=groupby_cols, columns='start', aggfunc=['mean', SD, SE, 'count'])
df = df.drop('year', axis=1)
# This aggregates accuracy grouped by dataset, model, history (index) and start (column)
df = df.pivot_table(index=groupby_cols, columns='start', aggfunc=['mean', SE])


# df['FWT'] = df['mean']['accuracy']['warm'] - df['mean']['accuracy']['cold']
# print(df)
df['FWT'] = fwt
df['SE'] *= 1.96 
df.rename({"SE": "1.96SE"}, inplace=True, axis=1)
print(df * 100) 

if args.save:
    print("Saving aggregated results to:", args.save)
    if args.latex:
        with open(args.save, 'w') as fhandle:
            print(df.to_latex(), file=fhandle)
    else:
        df.to_csv(args.save)
