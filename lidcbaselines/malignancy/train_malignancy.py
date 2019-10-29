"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from lidcbaselines.malignancy.malignancy import LIDCBaseline
from pathlib import Path
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def main(hparams, gpu_ids=None):
    # if gpu argument is passed, take it as the hparam gpu
    if gpu_ids:
        hparams.gpus = gpu_ids

    tt_logger = TestTubeLogger(
        save_dir=".",
        name="ntrain",
        version=f"ntr{hparams.trn_nb}-seed{hparams.split_seed}",
        debug=False,
        create_git_tag=False
    )
    log_path = Path(tt_logger.save_dir, tt_logger.name, f"version_{tt_logger.version}")

    checkpoint_callback = ModelCheckpoint(
        filepath=log_path / 'checkpoints',
        save_best_only=True,
        verbose=False,
        monitor='avg_val_loss',
        mode='min',
        prefix=''
    )
    
    # check if experiment already exists
    if log_path.exists():
        print(f"experiment {tt_logger.name} version {tt_logger.version} already exists, skipping")
        return 
    
    # init module
    model = LIDCBaseline(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
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
