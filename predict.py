import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import hydra
from hydra.utils import get_original_cwd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src.dataset import get_dataset
from src.model import TransformerRegressor, VectorAutoRegressor, LSTMRegressor
from src.utils import regression_metrics



@hydra.main(config_path="config", config_name="config")
def main(cfg):
    model_path = os.path.join(get_original_cwd(), cfg.predict.model)
    print(model_path)
    assert os.path.isfile(model_path), "you must specify a model with predict.model=<path-to-model>"
    
    input_window = cfg.dataset.window - 1
    cfg.dataset.window = cfg.dataset.window + cfg.predict.window

    dataset = get_dataset(cfg, cfg.predict.split)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.predict.batch_size,
        num_workers=os.cpu_count()
    )
    if cfg.architecture == 'TransformerRegressor':
        model = TransformerRegressor.load_from_checkpoint(model_path)
    elif cfg.architecture == 'VectorAutoRegressor':
        model = VectorAutoRegressor.load_from_checkpoint(model_path)
    elif cfg.architecture == 'LSTMRegressor':
        model = LSTMRegressor.load_from_checkpoint(model_path)
    model.eval()

    batch = next(iter(dataloader))
    all_predictions = []
    input_batch = batch[:, :input_window]
    for i in trange(0, cfg.predict.window):
        input_nan_mask = torch.isnan(input_batch)
        input_batch[input_nan_mask] = model.missing_placeholder

        predictions = model(input_batch)[:, -1, :].unsqueeze(1)
        input_batch = torch.cat((input_batch, predictions), dim=1)[:, 1:, :]
        all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions, dim=1)

    output_folder = os.path.join(cfg.predict.output_folder, cfg.architecture, cfg.dataset.name, cfg.predict.split)
    
    for b in trange(0, cfg.predict.batch_size, position=0, leave=False):
        batch_output_folder = os.path.join(output_folder, str(b))
        os.makedirs(batch_output_folder, exist_ok=True)

        inputs = batch[b, :input_window]
        predictions = all_predictions[b]
        actuals = batch[b, input_window:-1]
        
        for channel_i, channel_name in tqdm(enumerate(cfg.dataset.channels.data), total=len(cfg.dataset.channels.data), position=1, leave=False):
            channel_inputs = inputs[:, channel_i].detach().cpu().numpy()
            channel_predictions = predictions[:, channel_i].detach().cpu().numpy()
            channel_actuals = actuals[:, channel_i].detach().cpu().numpy()
            
            plt.figure()
            plt.title(channel_name)

            xi = len(channel_inputs)
            xp = len(channel_inputs) + len(channel_predictions)

            plt.plot(range(0, xi), channel_inputs, c="k")
            plt.plot(range(xi, xp), channel_predictions, c="r")
            plt.plot(range(xi, xp), channel_actuals, c="b")

            plt.savefig(os.path.join(batch_output_folder, f"{channel_name}.png"))
            plt.close()

            padded_inputs = np.zeros(xp)
            padded_predictions = np.zeros(xp)
            padded_actuals = np.zeros(xp)
            
            padded_inputs[:xi] = channel_inputs
            padded_predictions[xi:xp] = channel_predictions
            padded_actuals[xi:xp] = channel_actuals
            
            pd.DataFrame(dict(
                inputs=padded_inputs,
                predictions=padded_predictions,
                actuals=padded_actuals
            )).to_csv(os.path.join(batch_output_folder, f"{channel_name}.csv"))
    

if __name__  == "__main__":
    main()