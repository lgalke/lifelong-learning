""" Module holding the logic for writing results to disk in csv format """

import os
import pandas as pd


# do not change this
# unless also changing CSVResultsWriter.add_result()
RESULT_COLS = ['dataset',
               'history',
               'label_rate',
               'inductive',
               'seed',
               'backend',
               'model',
               'variant',
               'n_hidden',
               'n_layers',
               'dropout',
               'sampling',
               'batch_size',
               'saint_coverage',
               'initial_epochs',
               'initial_lr',
               'initial_wd',
               'annual_epochs',
               'annual_lr',
               'annual_wd',
               'start',
               'decay',
               'year',
               'epoch',
               'f1_macro',
               'accuracy',
               'open_learning',
               'doc_threshold',
               'doc_reduce_risk',
               'doc_alpha',
               'doc_class_weights',
               'open_tp',
               'open_tn',
               'open_fp',
               'open_fn',
               'open_mcc',
               'open_f1_macro']


def appendDFToCSV_void(df, csvFilePath, sep=","):
    """ Safe appending of a pandas df to csv file
    Source: https://stackoverflow.com/questions/17134942/pandas-dataframe-output-end-of-csv
    """
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception(
            "Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
                len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


class CSVResultsWriter:
    def __init__(self, config, columns=RESULT_COLS):
        self.config = config
        self.data = pd.DataFrame(columns=columns)

    def add_result(self, scores):
        self.data = self.data.append(
            pd.DataFrame(
                [[self.config.dataset,
                  self.config.history,
                  self.config.label_rate,
                  self.config.inductive,
                  self.config.seed,
                  self.config.backend,
                  self.config.model,
                  self.config.variant,
                  self.config.n_hidden,
                  self.config.n_layers,
                  self.config.dropout,
                  self.config.sampling,
                  self.config.batch_size,
                  self.config.saint_coverage,
                  self.config.initial_epochs,
                  self.config.lr,
                  self.config.weight_decay,
                  self.config.annual_epochs,
                  self.config.lr * self.config.rescale_lr,
                  self.config.weight_decay * self.config.rescale_wd,
                  self.config.start,
                  self.config.decay,
                  scores['task'],
                  scores['epoch'],
                  scores['f1_macro'],
                  scores['accuracy'],
                  self.config.open_learning,
                  self.config.doc_threshold,
                  self.config.doc_reduce_risk,
                  self.config.doc_alpha,
                  self.config.doc_class_weights,
                  scores.get('open_tp', 0),  # Optional
                  scores.get('open_tn', 0),
                  scores.get('open_fp', 0),
                  scores.get('open_fn', 0),
                  scores.get('open_mcc', 0),
                  scores.get('open_f1_macro', 0)
                  ]],
                columns=RESULT_COLS),
            ignore_index=True)

    def write(self, path):
        appendDFToCSV_void(self.data, path)
