import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import hydra
from hydra.utils import get_original_cwd
from tqdm import trange
import flatdict

from src.dataset import get_dataset
from src.model import TransformerRegressor
from src.utils import regression_metrics


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    model_path = os.path.join(get_original_cwd(), cfg.evaluate.model)
    assert os.path.isfile(model_path), "you must specify a model with evaluate.model=<path-to-model>"
    
    dataset = get_dataset(cfg, cfg.evaluate.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.evaluate.batch_size,
        num_workers=os.cpu_count()
    )

    model = TransformerRegressor.load_from_checkpoint(model_path)
    model.eval()

    results = list()

    for batch, _ in zip(dataloader, trange(0, cfg.evaluate.batches)):
        input_batch = batch[:, :-1]
        target_batch = batch[:, 1:]

        input_nan_mask = torch.isnan(input_batch)
        target_nan_mask = torch.isnan(target_batch)

        input_batch[input_nan_mask] = model.missing_placeholder
        predictions = model(input_batch)

        last_metrics = model.compute_metrics(
            trues=target_batch,
            predictions=predictions,
            metrics_fn=regression_metrics,
            step=-1
        )

        results.append(flatdict.FlatDict(last_metrics, delimiter="."))

    results = pd.DataFrame(results)
    results = results.mean(axis=0)
    print(results)
    
if __name__  == "__main__":
    main()