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
from lidcbaselines.modules import resnet18, Flatten, auc, accuracy
from lidcbaselines.dataloader import NoduleDataset
from sklearn.metrics import accuracy_score, roc_auc_score

import pytorch_lightning as pl


class LIDCBaseline(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # grab data and split in train / valid
        df = pd.read_csv(hparams.data_root / 'nodulelabels.csv')
        df['label']   = df['malignancy_binary']
        df['filedir'] = hparams.data_root

        rs = np.random.RandomState(seed=hparams.split_seed)
        train_ids = rs.rand(len(df)) < 0.8
        self.dftrain = df[train_ids]
        self.dfvalid = df[~train_ids]

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
        val_metrics  = {'auc': auc_val, 'accuracy': accuracy_val}
        val_metrics.update({'avg_val_loss': avg_loss})

        return val_metrics

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def tng_dataloader(self):
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
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--split_seed', default=123, type=int)
        parser.add_argument('--data_root', default=Path(root_dir) / 'data', type=Path)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=1000, type=int)

        return parser

