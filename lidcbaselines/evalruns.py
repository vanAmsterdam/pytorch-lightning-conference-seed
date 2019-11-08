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
        try:
            mdf = pd.read_csv(vpath / 'metrics.csv')
        except Exception as e:
            print(f"error loading file {vpath / 'metrics.csv'}: {e}")
        mdf['version']   = vname
        metricdfs[vname] = mdf
    metricdf  = pd.concat(metricdfs)
    metricdf['ntr']  = metricdf.version.apply(lambda x: re.search(r'(?<=ntr)(\d+)', x).group())
    metricdf['seed'] = metricdf.version.apply(lambda x: re.search(r'(?<=seed)(\d+)', x).group())
    if 'outcome' in metricdf.version.iloc[0]:
        metricdf['outcome'] = metricdf.version.apply(lambda x: re.search(r'(?<=outcome)(.*)(?=-ntr)', x).group())
    metricdf = metricdf.join((metricdf.groupby('version').epoch.max()==metricdf.epoch.max()).rename('completed'), on='version')
    metricdf.to_csv(expdir / 'allmetrics.csv', index=False)

    # aggregate metrics by version
    metricnames = args.metrics.split(',')
    minmetrics  = args.minmetrics.split(',')
    byrundfs    = {}
    bytypedfs   = {}
    for metric in metricnames:
        groupvars = ['version', 'ntr', 'seed', 'completed']
        if 'outcome' in metricdf.columns:
            groupvars += ['outcome']
        aggops    = ['count', 'mean', 'std', 'min', 'max']
        if metric in minmetrics:
            df = metricdf.groupby(groupvars)[metric].min()
        else:
            df = metricdf.groupby(groupvars)[metric].max()
        byrundfs[metric] = df

        # filter on completed runs only
        # runcomplete = metricdf.groupby([x for x in groupvars if x not in ['completed']]).completed.max()
        # df = df[runcomplete]

        # aggregate by type
        df = df.groupby('ntr').agg(aggops)
        df['ntr_int'] = df.index.astype(int)
        df.sort_values(axis=0, by=['ntr_int'], ascending=False, inplace=True)
        bytypedfs[metric] = df

    metricsbyrun = pd.concat(byrundfs, axis=1)
    metricsbyrun.columns = [x+'_min' if x in minmetrics else x+'_max' for x in metricsbyrun.columns]
    aggmetrics   = pd.concat(bytypedfs)
    print(metricsbyrun)
    print(aggmetrics)
    metricsbyrun.to_csv(expdir / 'metricsbyrun.csv', index=True)
    aggmetrics.to_csv(expdir / 'aggmetrics.csv', index=True)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)