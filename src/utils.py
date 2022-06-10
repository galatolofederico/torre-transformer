import sklearn.metrics
import scipy.stats
import numpy as np
from omegaconf import DictConfig, OmegaConf
import flatdict
import torch
import io
import PIL
import math

def confidence_interval(arr, confidence=.95):
    if np.isnan(arr).any(): print("Warning skipping some NaNs")
    data = arr[~np.isnan(arr)]
    se = scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return h

def regression_metrics(y_true, y_pred):

    rdp_vector = np.abs(y_true - y_pred)/((np.abs(y_true) + np.abs(y_pred))/2)

    return dict(
        chan_mean = np.mean(y_true),
        pred_mean = np.mean(y_pred),
        chan_std = y_true.std(),
        pred_std = y_pred.std(),
        mean_rpd = np.mean(rdp_vector),
        std_rpd = rdp_vector.std(),
        mean_absolute_error = float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
        mean_squared_error = float(sklearn.metrics.mean_squared_error(y_true, y_pred)),
        d2_tweedie_score = float(sklearn.metrics.d2_tweedie_score(y_true, y_pred)),
        r2_score = float(sklearn.metrics.r2_score(y_true, y_pred)),
        explained_variance_score = float(sklearn.metrics.explained_variance_score(y_true, y_pred))
    )

def hp_from_cfg(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(flatdict.FlatDict(cfg, delimiter="/"))


def deep_move(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [deep_move(d, device) for d in data]
    elif isinstance(data, tuple):
        return (deep_move(d, device) for d in data)
    elif isinstance(data, dict):
        return {k: deep_move(v, device) for k, v in data.items()}
    else:
        raise TypeError(
            f"Data structure of type {type(data)} cannot be moved to {device}"
        )

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)