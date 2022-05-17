import torch
import torch.nn.functional as F
import flatdict
from torch.utils.data import DataLoader
import pytorch_lightning
from x_transformers.x_transformers import AttentionLayers, AbsolutePositionalEmbedding
import hydra
import os
from einops import rearrange

from src.dataset import get_dataset
from src.utils import regression_metrics


class TowerModel(pytorch_lightning.LightningModule):
    def __init__(
            self,
            input_channels,
            channel_names,
            seq_len,
            lr,
            log_metrics_each
        ):
        super(TowerModel, self).__init__()
        self.missing_placeholder = torch.nn.Parameter(torch.randn(1))
        self.save_hyperparameters()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.lr = lr
        self.channel_names = channel_names
        self.log_metrics_each = log_metrics_each
        self.training_step = lambda batch, batch_nb: self.step("train", batch, batch_nb)
        self.validation_step = lambda batch, batch_nb: self.step("validation", batch, batch_nb)

    def compute_metrics(self, trues, predictions, metrics_fn, step=-1):
        ret = dict(mean=dict())

        for channel_i, channel_name in enumerate(self.channel_names):
            channel_trues = trues[:, step, channel_i]
            channel_predictions = predictions[:, step, channel_i]
            channel_mask = ~torch.isnan(channel_trues)
            
            channel_trues = channel_trues[channel_mask]
            channel_predictions = channel_predictions[channel_mask]
            
            channel_metrics = metrics_fn(channel_trues.detach().cpu().numpy(), channel_predictions.detach().cpu().numpy())
            ret[channel_name] = channel_metrics

        for metric_name in channel_metrics.keys():
            metric_values = []
            for channel_name in self.channel_names:
                metric_values.append(ret[channel_name][metric_name])
            ret["mean"][metric_name] = sum(metric_values) / len(metric_values)
        
        return ret

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
        
        self.log(f"{step}/loss", loss.item(), prog_bar=True)
        
        last_metrics = self.compute_metrics(
            trues=target_batch,
            predictions=predictions,
            metrics_fn=regression_metrics,
            step=-1
        )

        log_metrics = {f"{step}/last": last_metrics}
        flat_metrics = flatdict.FlatDict(log_metrics, delimiter="/")
        self.log_dict(flat_metrics)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)




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



class TransformerRegressor(TowerModel):
    def __init__(
        self,
        transformer_decoder_dim,
        transformer_decoder_depth,
        transformer_decoder_heads,
        transformer_decoder_dropout,
        input_channels,
        channel_names,
        seq_len,
        lr,
        log_metrics_each
    ):
        super(TransformerRegressor, self).__init__(
            input_channels,
            channel_names,
            seq_len,
            lr,
            log_metrics_each
        )

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

    def compute_metrics(self, trues, predictions, metrics_fn, step=-1):
        ret = dict(mean=dict())

        for channel_i, channel_name in enumerate(self.channel_names):
            channel_trues = trues[:, step, channel_i]
            channel_predictions = predictions[:, step, channel_i]
            channel_mask = ~torch.isnan(channel_trues)
            
            channel_trues = channel_trues[channel_mask]
            channel_predictions = channel_predictions[channel_mask]
            
            channel_metrics = metrics_fn(channel_trues.detach().cpu().numpy(), channel_predictions.detach().cpu().numpy())
            ret[channel_name] = channel_metrics

        for metric_name in channel_metrics.keys():
            metric_values = []
            for channel_name in self.channel_names:
                metric_values.append(ret[channel_name][metric_name])
            ret["mean"][metric_name] = sum(metric_values) / len(metric_values)
        
        return ret

    def forward(self, X):
        embeddings = self.input_embedding(X)
        encodings = self.decoder(embeddings)
        predictions = self.regression_head(encodings)

        return predictions



class VectorAutoRegressor(TowerModel):
    def __init__(
        self,
        input_channels,
        channel_names,
        seq_len,
        lr,
        log_metrics_each
    ):
        super(VectorAutoRegressor, self).__init__(
            input_channels,
            channel_names,
            seq_len,
            lr,
            log_metrics_each
        )

        input_flatted_len = input_channels * (seq_len - 1)
        self.layer1 = torch.nn.Linear(input_flatted_len, input_flatted_len)

        mask = torch.ones(input_channels, seq_len-1, seq_len-1, dtype=int)
        mask = torch.block_diag(*mask)
        tmp = torch.ones(input_flatted_len, input_flatted_len, dtype=int)
        self.mask = (mask | tmp.tril()).float()

        self.layers = torch.nn.ModuleList()

        for i in range(0, 71):
            self.layers.append(torch.nn.Linear(31*(i+1), 31))
        
    

    def forward(self, X):
        outputs = []
        for i, layer in enumerate(self.layers):
            Xi = rearrange(X[:, :(i+1), :], "b t d -> b (t d)")
            outputs.append(layer(Xi))
        outputs = torch.stack(outputs, dim=1)

        return outputs
            


class LSTMRegressor(TowerModel):
    def __init__(
        self,
        hidden_size, 
        num_layers,
        input_channels,
        channel_names,
        seq_len,
        lr,
        log_metrics_each
    ):

        super(LSTMRegressor, self).__init__(
            input_channels,
            channel_names,
            seq_len,
            lr,
            log_metrics_each
        )
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(
            input_size=input_channels, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True
        )

        self.linear = torch.nn.Linear(hidden_size, input_channels)
        
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        output, (h_out, _) = self.lstm(x, (h_0, c_0))
        pred = self.linear(output)

        return pred




@hydra.main(config_path=None, config_name="config")
def main(cfg):
    ds = get_dataset(cfg, "train")
    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        num_workers=os.cpu_count()
    )

    model_2 = LSTMRegressor( 
        hidden_size = cfg.model.lstm.hidden_size, 
        num_layers = cfg.model.lstm.num_layers,
        channel_names=ds.channel_names,
        input_channels=len(cfg.dataset.channels.data),
        seq_len=cfg.dataset.window,
        lr=cfg.train.lr,
        log_metrics_each=cfg.log.metrics_each)

    for i, elem in enumerate(dl):
        loss = model_2.training_step(elem, i)
        loss.backward()
        print(loss)



'''
@hydra.main(config_path=None, config_name="config")
def main(cfg):
    ds = get_dataset(cfg, "train")
    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        num_workers=os.cpu_count()
    )

    model_1 = TransformerRegressor(
        transformer_decoder_dim=cfg.model.transformer.decoder_dim,
        transformer_decoder_depth=cfg.model.transformer.decoder_depth,
        transformer_decoder_heads=cfg.model.transformer.decoder_heads,
        transformer_decoder_dropout=cfg.model.transformer.decoder_dropout,
        channel_names=ds.channel_names,
        input_channels=len(cfg.dataset.channels.data),
        seq_len=cfg.dataset.window,
        lr=cfg.train.lr,
        log_metrics_each=cfg.log.metrics_each
    )

    model_2 = VectorAutoRegressor(
        channel_names=ds.channel_names,
        input_channels=len(cfg.dataset.channels.data),
        seq_len=cfg.dataset.window,
        lr=cfg.train.lr,
        log_metrics_each=cfg.log.metrics_each
    )

    for i, elem in enumerate(dl):
        loss = model_2.training_step(elem, i)
        loss.backward()
        print(loss)
'''        

if __name__  == "__main__":
    main()