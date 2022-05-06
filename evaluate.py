import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import hydra
from hydra.utils import get_original_cwd
from tqdm import trange
import flatdict

from src.dataset import get_dataset
from src.model import TransformerRegressor, VectorAutoRegressor
from src.utils import regression_metrics


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    model_path = os.path.join(get_original_cwd(), cfg.evaluate.model)
    print(model_path)
    assert os.path.isfile(model_path), "you must specify a model with evaluate.model=<path-to-model>"
    
    dataset = get_dataset(cfg, cfg.evaluate.split)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.evaluate.batch_size,
        num_workers=os.cpu_count()
    )
    if cfg.architecture == 'TransformerRegressor':
        model = TransformerRegressor.load_from_checkpoint(model_path)
    elif cfg.architecture == 'VectorAutoRegressor':
        model = VectorAutoRegressor.load_from_checkpoint(model_path)
    model.eval()

    results = dict()
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

        for channel, metrics in last_metrics.items():
            if channel not in results: results[channel] = dict()
            for metric, value in metrics.items():
                if metric not in results[channel]: results[channel][metric] = list()
                results[channel][metric].append(value)

    results = pd.DataFrame(results)
    mean_results = results.applymap(lambda e: np.array(e).mean())
    std_results = results.applymap(lambda e: np.array(e).std())

    output_folder = os.path.join(cfg.evaluate.output_folder, cfg.architecture, cfg.dataset.name, cfg.evaluate.split) 
    os.makedirs(output_folder, exist_ok=True)
    
    mean_results.to_csv(os.path.join(output_folder, "mean.csv"))
    std_results.to_csv(os.path.join(output_folder, "std.csv"))
    
    print(mean_results)
    

if __name__  == "__main__":
    main()