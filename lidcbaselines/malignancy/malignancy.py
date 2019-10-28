"""
This file defines the core research contribution   
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
from lidcbaselines.modules import resnet18, Flatten
from lidcbaselines.dataloader import NoduleDataset
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve
from test_tube import HyperOptArgumentParser
import pytorch_lightning as pl


class LIDCBaseline(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # grab data and split in train / valid
        df = pd.read_csv(hparams.data_root / 'nodulelabels.csv')
        df['label']   = df['malignancy_binary']
        df['filedir'] = hparams.data_root

        # determine train / valid split
        rs        = np.random.RandomState(seed=hparams.split_seed)
        idxs      = np.arange(len(df))
        rs.shuffle(idxs)
        num_train = int(.8*len(df))
        train_ids = idxs[:num_train]
        valid_ids = idxs[num_train:]

        # possibly subsample train
        if hparams.trn_subsample_pct < 1.0:
            train_ids = train_ids[:int(hparams.trn_subsample_pct*num_train)]
        if hparams.trn_nb > 0:
            train_ids = train_ids[:hparams.trn_nb]
        if hparams.trn_subsample_pct < 1.0 and hparams.trn_nb > 0:
            raise ValueError(f'please specify only trn_subsample_pct or trn_nb, they are: {hparams.trn_subsample_pct} and {hparams.trn_nb}')

        self.dftrain = df.iloc[train_ids]
        self.dfvalid = df.iloc[valid_ids]

        # build model
        self.__build_model()

    def __build_model(self):
        self.encoder = resnet18(pretrainedpath='/home/wamsterd/git/lc-survival/imaging/resnet18-kinetics.pth', encode_only=True)
        self.flatten = Flatten()
        self.fc      = nn.Linear(512, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def loss(self, pred, label):
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(pred.squeeze(), label.squeeze().float())

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y     = batch
        y_hat    = self.forward(x)
        loss_val = self.loss(y_hat, y)
        return {'loss': loss_val}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y     = batch
        y_hat    = self.forward(x)
        loss_val = self.loss(y_hat, y)

        return {'val_loss': loss_val, 
                'y_hat': y_hat.view(-1,1),
                'y': y.view(-1,1)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y_hats   = torch.cat([x['y_hat'] for x in outputs], dim=0)
        ys       = torch.cat([x['y'] for x in outputs], dim=0)

        # calculate some metrics
        auc_val      = roc_auc_score(ys.cpu().long(), y_hats.cpu())
        accuracy_val = accuracy_score(ys.cpu().long(), (y_hats.cpu()>0.0).long())
        precision, recall, thresholds = precision_recall_curve(ys.cpu().long(), y_hats.cpu())
        prauc_val    = auc(recall, precision) # put recall as first argument because this argument needs to be sorted
        val_metrics  = {'auc': auc_val, 'accuracy': accuracy_val, 'prauc': prauc_val}
        val_metrics.update({'avg_val_loss': avg_loss})

        # possibly monitor preds
        if self.hparams.monitor_preds:
            self.experiment.add_histogram('preds', y_hats, self.global_step)

        return val_metrics

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(NoduleDataset(self.dftrain, self.hparams, split='train'), 
                          batch_size=self.hparams.batch_size,
                          shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(NoduleDataset(self.dfvalid, self.hparams, split='valid'), 
                          batch_size=self.hparams.batch_size,
                          shuffle=False)


    # @pl.data_loader
    # def test_dataloader(self):
        # OPTIONAL
        # return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = HyperOptArgumentParser(strategy='grid_search', parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.01, type=float)
        parser.add_argument('--batch_size', default=128, type=int)
        # parser.add_argument('--split_seed', default=123, type=int)
        parser.opt_list('--split_seed', default=1, type=int, tunable=True,
                        options=list(1000+np.arange(20).astype(np.int32)))
                        # options=list(1000+np.arange(2).astype(np.int32)))
        parser.add_argument('--data_root', default=Path(root_dir) / 'data', type=Path)
        parser.add_argument('--monitor_preds', action='store_true', default=False, help='export histograms of preds')
        parser.add_argument('--trn_subsample_pct', default=1.0, type=float, help='subsample percentage of training data')
        parser.opt_list('--trn_nb', default=0, type=int, help='number of training samples (0 = take all samples)', tunable=True,
                        options=[2000, 1750, 1500, 1250, 1000, 750, 500])
                        # options=[750, 500])
        # parser.add_argument('--trn_nb', default=0, type=int, help='number of training samples (0 = take all samples)')

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=200, type=int)

        return parser

