import argparse
import os
from torch.utils.data import DataLoader
import pytorch_lightning
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import hydra

from src.dataset import get_dataset
from src.model import TransformerRegressor
from src.utils import hp_from_cfg

@hydra.main(config_path="config", config_name="config")
def train(cfg):
    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    
    seed_everything(cfg.train.seed)
    
    loggers = list()
    callbacks = list()
    if cfg.train.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)

    last_checkpoint_callback = ModelCheckpoint(
        filename="last",
        save_last=True,
    )
    min_loss_checkpoint_callback = ModelCheckpoint(
        monitor="train/loss",
        filename="min-loss",
        save_top_k=1,
        mode="min",
    )

    callbacks.extend([
        last_checkpoint_callback,
        min_loss_checkpoint_callback
    ])
    
    train_dataset = get_dataset(cfg, "train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=os.cpu_count()
    )

    model = TransformerRegressor(
        transformer_decoder_dim=cfg.model.transformer.decoder_dim,
        transformer_decoder_depth=cfg.model.transformer.decoder_depth,
        transformer_decoder_heads=cfg.model.transformer.decoder_heads,
        transformer_decoder_dropout=cfg.model.transformer.decoder_dropout,
        channel_names=train_dataset.channel_names,
        input_channels=len(cfg.dataset.channels.data),
        seq_len=cfg.dataset.window,
        lr=cfg.train.lr,
        log_metrics_each=cfg.log.metrics_each
    )

    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        gpus=cfg.train.gpus,
        log_every_n_steps=1,
        #val_check_interval=cfg.train.validation_interval,
        #limit_val_batches=cfg.train.validation_batches
    )
    
    trainer.fit(model, train_dataloader)

if __name__ == "__main__":
    train()
