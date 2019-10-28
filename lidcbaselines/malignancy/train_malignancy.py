"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from lidcbaselines.malignancy.malignancy import LIDCBaseline
from pathlib import Path
from pytorch_lightning.logging import TestTubeLogger

def main(hparams, gpu_ids=None):
    # init module
    model = LIDCBaseline(hparams)

    # if gpu argument is passed, take it as the hparam gpu
    if gpu_ids:
        hparams.gpus = gpu_ids

    # generate experiment
    if hparams.version == '':
        hparams.version = 0.1

    tt_logger = TestTubeLogger(
        save_dir=".",
        name="experiments",
        debug=False,
        create_git_tag=False
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(
        logger = tt_logger,
        # experiment = exp,
        max_nb_epochs=hparams.max_nb_epochs,
        # gpus=hparams.gpus,
        gpus=gpu_ids,
        nb_gpu_nodes=hparams.nodes,
        nb_sanity_val_steps=2
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--version', type=str, default='')

    # define root
    root_dir = Path('/home/wamsterd/git/lidcbaselines/lidcbaselines')

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = LIDCBaseline.add_model_specific_args(parser, root_dir)

    # parse params
    hparams = parser.parse_args()
    
    # run trial(s)
    # hparams.optimize_parallel_gpu(main, hparams.gpus)
    hparams.optimize_parallel_gpu(main, ['0'])
