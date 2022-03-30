import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning
from x_transformers.x_transformers import AttentionLayers, AbsolutePositionalEmbedding
import hydra
import os

from src.dataset import get_dataset

class Transformer(torch.nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            causal=False,
            dropout=0,
            seq_len=None,
            pos_embedding=False,
            cross_attend=False
        ):
        super(Transformer, self).__init__()

        self.attn_layers = AttentionLayers(
            dim = dim,
            depth = depth,
            heads = heads,
            cross_attend = cross_attend,
            causal = causal
        )

        self.pos_embedding = None
        if pos_embedding:
            assert seq_len is not None, "Must specify seq_len when using positional embeddings"
            self.pos_embedding = AbsolutePositionalEmbedding(dim, seq_len) 
        self.norm = torch.nn.LayerNorm(self.attn_layers.dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, return_attention=False, **kwargs):
        if self.pos_embedding is not None:
            embeddings = embeddings + self.pos_embedding(embeddings)
        
        embeddings = self.dropout(embeddings)
        latent, intermediates = self.attn_layers(embeddings, return_hiddens=True, **kwargs)
        latent = self.norm(latent)

        if return_attention:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return latent, attn_maps
        
        return latent


class TransformerRegressor(pytorch_lightning.LightningModule):
    def __init__(
        self,
        transformer_decoder_dim,
        transformer_decoder_depth,
        transformer_decoder_heads,
        transformer_decoder_dropout,
        input_channels,
        seq_len,
        lr
    ):
        super(TransformerRegressor, self).__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.decoder = Transformer(
            dim = transformer_decoder_dim,
            depth = transformer_decoder_depth,
            heads = transformer_decoder_heads,
            dropout = transformer_decoder_dropout,
            seq_len = seq_len,
            pos_embedding = True,
            cross_attend = False,
            causal = True
        )

        self.input_embedding = torch.nn.Linear(input_channels, transformer_decoder_dim)
        self.regression_head = torch.nn.Linear(transformer_decoder_dim, input_channels)

        self.missing_placeholder = torch.nn.Parameter(torch.randn(1))

        self.training_step = lambda batch, batch_nb: self.step("train", batch, batch_nb)
        self.validation_step = lambda batch, batch_nb: self.step("validation", batch, batch_nb)

    def forward(self, X):
        embeddings = self.input_embedding(X)
        encodings = self.decoder(embeddings)
        predictions = self.regression_head(encodings)

        return predictions
    
    def step(self, step, batch, batch_nb):
        input_batch = batch[:, :-1]
        target_batch = batch[:, 1:]

        input_nan_mask = torch.isnan(input_batch)
        target_nan_mask = torch.isnan(target_batch)

        input_batch[input_nan_mask] = self.missing_placeholder
        predictions = self(input_batch)
        
        loss_predictions = predictions[~target_nan_mask]
        loss_targets = target_batch[~target_nan_mask]

        loss = F.mse_loss(loss_predictions, loss_targets)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)



@hydra.main(config_path=None, config_name="config")
def main(cfg):
    ds = get_dataset(cfg, "train")
    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        num_workers=os.cpu_count()
    )

    model = TransformerRegressor(
        transformer_decoder_dim=cfg.model.transformer.decoder_dim,
        transformer_decoder_depth=cfg.model.transformer.decoder_depth,
        transformer_decoder_heads=cfg.model.transformer.decoder_heads,
        transformer_decoder_dropout=cfg.model.transformer.decoder_dropout,
        input_channels=len(cfg.dataset.channels.data),
        seq_len=cfg.dataset.window,
        lr=cfg.train.lr,
    )

    for i, elem in enumerate(dl):
        loss = model.training_step(elem, i)
        loss.backward()
        print(loss)
    
if __name__  == "__main__":
    main()