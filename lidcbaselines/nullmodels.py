'''
script to create null models for baseline stats
'''
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve, log_loss
from argparse import ArgumentParser
import torch
from torch.nn import BCELoss

df = pd.read_csv(Path('data') / 'nodulelabels.csv')
binary_outcomes = [x for x in df.columns if '_binary' in x]

mdfs = {}
for outcome in binary_outcomes:
    y     = df[outcome]
    X     = np.zeros_like(y).reshape(-1,1)
    lr    = LogisticRegression().fit(X,y)
    pred_probs = lr.predict_proba(X)[:,0]
    pred_y     = lr.predict(X)
    
    precision, recall, thresholds = precision_recall_curve(y, pred_y)
    prauc_val    = auc(recall, precision) # put recall as first argument because this argument needs to be sorted

    criterion  = BCELoss()

    metrics = {
        'auc':      [roc_auc_score(y, pred_probs)],
        'accuracy': [accuracy_score(y, pred_y)],
        'prauc':    [prauc_val],
        'avg_val_loss': [log_loss(y, pred_y)],
        'avg_val_loss2': [criterion(torch.tensor(pred_y).float(), torch.tensor(y).float())]
    }
    mdf = pd.DataFrame.from_dict(metrics, orient='columns')
    mdfs[outcome.split('_')[0]] = mdf

mdf = pd.concat(mdfs, ignore_index=False)
print(mdf)
mdf.to_csv('nullmetrics.csv', index=True)