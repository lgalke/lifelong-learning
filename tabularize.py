import argparse
import math
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("results_csv_file", help="Path to results file")
parser.add_argument("-g", "--groupby", nargs='+', help="Group on this variable",
                    default=['model', 'annual_lr'])
parser.add_argument("-y", "--score", help="Score (target) column",
                    default='accuracy')
parser.add_argument("--latex", default=False, action='store_true', help="Produce latex output")
parser.add_argument("--save", help="Path to save resulting table")
args = parser.parse_args()
print("Loading data:", args.results_csv_file)
df = pd.read_csv(args.results_csv_file)
print("N =", len(df))
print("Grouping by:", args.groupby)



if args.score == 'global_mcc':
    # Compute Global MCC per seed first
    # groups = df.groupby(args.groupby + ['seed'])
    groups = df.groupby(args.groupby)

    tp = groups['open_tp'].sum().astype(np.float64)
    tn = groups['open_tn'].sum().astype(np.float64)
    fp = groups['open_fp'].sum().astype(np.float64)
    fn = groups['open_fn'].sum().astype(np.float64)
    print('tp', tp)
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)

    # Alternate computation, avoiding NaNs

    # PPV = tp / (tp + fp)                    # <-- NaN with t=0
    # TPR = tp / (tp + fn)
    # TNR = tn / (tn + fp)
    # NPV = tn / (tn + fn)

    # FDR = fp / (fp + tp)  # 1 - TNR         # <-- NaN with t=0
    # FNR = fn / (fn + tp)  # 1 - TPR
    # FPR = fp / (fp + tn)  # 1 - TNR
    # FOR = fn / (fn + tn)  # 1 - NPV

    # print("PPV", PPV)
    # print("TPR", TPR)
    # print("TNR", TNR)
    # print("NPV", NPV)

    # print("FDR", FDR)
    # print("FNR", FNR)
    # print("FPR", FPR)
    # print("FOR", FOR)

    # global_mcc = np.sqrt(PPV) * np.sqrt(TPR) * np.sqrt(TNR) * np.sqrt(NPV)\
    #     - np.sqrt(FDR) * np.sqrt(FNR) * np.sqrt(FPR) * np.sqrt(FOR)



    nominator = tp * tn - fp * fn
    # print("nom", nominator)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    # print("denom", denominator)
    global_mcc = nominator / np.sqrt(denominator)

    # results = global_mcc.groupby(args.groupby).mean()
    results = global_mcc

else:
    groups = df.groupby(args.groupby)
    results = pd.DataFrame(groups[args.score].mean())
    results['SD'] = groups[args.score].std()
    results['SE'] = groups[args.score].std() / np.sqrt(groups[args.score].count())
    results[args.score+'-ci95'] = results[args.score].map('{:.4f}'.format) + "+-" + (1.96 * results['SE']).map('{:.2f}'.format)
    results.drop([args.score, 'SD', 'SE'], axis=1, inplace=True)

print(results)
if args.latex:
    print(results.to_latex())
if args.save:
    print("Saving aggregated results to:", args.save)
    if args.latex:
        with open(args.save, 'w') as fhandle:
            print(results.to_latex(), file=fhandle)
    else:
        results.to_csv(args.save)
