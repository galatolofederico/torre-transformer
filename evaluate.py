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
from src.model import TransformerRegressor, VectorAutoRegressor, LSTMRegressor
from src.utils import regression_metrics, confidence_interval


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
    elif cfg.architecture == 'LSTMRegressor':
        model = LSTMRegressor.load_from_checkpoint(model_path)
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
    ci_results = results.applymap(lambda e: confidence_interval(np.array(e)))

    output_folder = os.path.join(cfg.evaluate.output_folder, cfg.architecture, cfg.dataset.name, cfg.evaluate.split) 
    os.makedirs(output_folder, exist_ok=True)
    
    mean_results.to_csv(os.path.join(output_folder, "mean.csv"))
    std_results.to_csv(os.path.join(output_folder, "std.csv"))
    ci_results.to_csv(os.path.join(output_folder, "ci.csv"))


    # metriche aggregate

    results_def = results.loc[:, results.columns.str.startswith('DEF')]
    results_tel = results.loc[:, results.columns.str.startswith('TEL')]
    results_def_tel = results.loc[:, results.columns.str.startswith('DEF') | results.columns.str.startswith('TEL')]

    mean_results_def = results_def.applymap(lambda e: np.array(e).mean())
    std_results_def = results_def.applymap(lambda e: np.array(e).std())
    ci_results_def = results_def.applymap(lambda e: confidence_interval(np.array(e)))

    mean_results_tel = results_tel.applymap(lambda e: np.array(e).mean())
    std_results_tel = results_tel.applymap(lambda e: np.array(e).std())
    ci_results_tel = results_tel.applymap(lambda e: confidence_interval(np.array(e)))

    mean_results_def_tel = results_def_tel.applymap(lambda e: np.array(e).mean())
    std_results_def_tel = results_def_tel.applymap(lambda e: np.array(e).std())
    ci_results_def_tel = results_def_tel.applymap(lambda e: confidence_interval(np.array(e)))

    output_folder = os.path.join(cfg.evaluate.output_folder, cfg.architecture, cfg.dataset.name, cfg.evaluate.split, 'aggregate_metric') 
    os.makedirs(output_folder, exist_ok=True)

    mean_results_def.to_csv(os.path.join(output_folder, "mean_def.csv"))
    std_results_def.to_csv(os.path.join(output_folder, "std_def.csv"))
    ci_results_def.to_csv(os.path.join(output_folder, "ci_def.csv"))

    mean_results_tel.to_csv(os.path.join(output_folder, "mean_tel.csv"))
    std_results_tel.to_csv(os.path.join(output_folder, "std_tel.csv"))
    ci_results_tel.to_csv(os.path.join(output_folder, "ci_tel.csv"))

    mean_results_def_tel.to_csv(os.path.join(output_folder, "mean_def_tel.csv"))
    std_results_def_tel.to_csv(os.path.join(output_folder, "std_def_tel.csv"))
    ci_results_def_tel.to_csv(os.path.join(output_folder, "ci_def_tel.csv"))

    

if __name__  == "__main__":
    main()