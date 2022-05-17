import torch
import random
import pandas as pd
import numpy as np
import hydra
from datetime import datetime
import yaml

class TowerDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        filename,
        window,
        date_channel,
        data_channels,
        date_ranges,
        seed=42,
        stats=None
    ):
        self.data = pd.read_csv(filename)
        self.data[date_channel] = pd.to_datetime(self.data[date_channel])
        self.data = self.data.set_index(date_channel)
        self.stats = None
        if stats is not None:
            self.stats = dict()
            for stat in stats:
                self.stats[stat["channel"]] = dict(
                    mean=stat["mean"],
                    std=stat["std"]
                )

        self.window = window
        
        for data_channel in data_channels: assert data_channel in self.data, f"Channel {data_channel} is missing in {filename}"
        self.data_channels = data_channels
        
        self.date_ranges = []
        for date_range in date_ranges:
            date_from = datetime.strptime(date_range["from"], "%d/%m/%Y")
            date_to = datetime.strptime(date_range["to"], "%d/%m/%Y")
            date_mask = (self.data.index > date_from) & (self.data.index < date_to) 
            assert len(self.data[date_mask]) > 0, f"No values in interval {date_from} - {date_to}"
            self.date_ranges.append(dict(
                mask=date_mask
            ))
        
        total_values = sum([len(self.data[date_range["mask"]]) for date_range in self.date_ranges])
        for i, date_range in enumerate(self.date_ranges):
            self.date_ranges[i]["weight"] = len(self.data[date_range["mask"]])/total_values

        self.rng = np.random.default_rng(seed)
    
    @property
    def channel_names(self):
        return self.data_channels

    def __iter__(self):
        return self

    def __next__(self):
        date_range = self.rng.choice(
            self.date_ranges,
            1,
            p=[date_range["weight"] for date_range in self.date_ranges]
        )[0]

        data = self.data[date_range["mask"]]
        i = self.rng.integers(0, len(data)-self.window)
        data = data.iloc[i:i+self.window]
        
        ret = torch.zeros(self.window, len(self.data_channels))
        for i, data_channel in enumerate(self.data_channels):
            if self.stats is not None and data_channel in self.stats:
                ret[:, i] = (torch.tensor(data[data_channel].to_numpy()) - self.stats[data_channel]["mean"])/self.stats[data_channel]["std"]
            else:
                ret[:, i] = torch.tensor(data[data_channel].to_numpy())
        
        return ret
        
def get_dataset(cfg, split, use_stats=True):
    stats = None
    if use_stats:
        with open(cfg.dataset.splits[split].stats, "r") as f:
            stats = yaml.safe_load(f)
    return TowerDataset(
        filename=cfg.dataset.filename,
        window=cfg.dataset.window,
        date_channel=cfg.dataset.channels.date,
        data_channels=cfg.dataset.channels.data,
        date_ranges=cfg.dataset.splits[split].dates,
        stats=stats
    )


@hydra.main(config_path=None, config_name="config")
def main(cfg):
    ds = get_dataset(cfg, "train")

    for elem in ds:
        print(elem.shape)

if __name__  == "__main__":
    main()