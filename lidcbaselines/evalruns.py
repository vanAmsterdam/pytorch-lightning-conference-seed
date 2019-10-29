'''
evaluate runs
'''

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--outcome',    default='malignancy', type=str, help='name of outcome / dir')
parser.add_argument('--experiment', default='ntrain', type=str, help='name of experiment (subdir of outcome)')
parser.add_argument('--metrics',    default='accuracy,auc,prauc,avg_val_loss', type=str, help='name of metrics, separated by a ,')
parser.add_argument('--minmetrics', default='avg_val_loss', type=str, help='name of metrics that should be as low as possible, separated by a ,')

def main(args):
    # find versions
    expdir       = Path(args.outcome, args.experiment)
    versionpaths = [x for x in list(expdir.glob('*')) if x.is_dir() and x.name.startswith('version')]
    print(f"found {len(versionpaths)} versions")
    versionnames = [x.name.split('version_')[1] for x in versionpaths]
    
    # gather all metrics
    metricdfs = {}
    for vname, vpath in zip(versionnames, versionpaths):
        mdf              = pd.read_csv(vpath / 'metrics.csv')
        mdf['version']   = vname
        metricdfs[vname] = mdf
    metricdf  = pd.concat(metricdfs)
    metricdf['ntr']  = metricdf.version.apply(lambda x: re.search(r'(?<=ntr)(\d+)', x).group())
    metricdf['seed'] = metricdf.version.apply(lambda x: re.search(r'(?<=seed)(\d+)', x).group())
    metricdf.to_csv(expdir / 'allmetrics.csv', index=False)

    # aggregate metrics by version
    metricnames = args.metrics.split(',')
    minmetrics  = args.minmetrics.split(',')
    aggdfs      = {}
    for metric in metricnames:
        if metric in minmetrics:
            aggdf = metricdf.groupby(['version', 'ntr'])[metric].min().groupby('ntr').agg(['count', 'min', 'std'])
        else:
            aggdf = metricdf.groupby(['version', 'ntr'])[metric].max().groupby('ntr').agg(['count', 'max', 'std'])
        aggdfs[metric] = aggdf
    aggmetrics = pd.concat(aggdfs)
    print(aggmetrics)
    aggmetrics.to_csv(expdir / 'aggmetrics.csv', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)