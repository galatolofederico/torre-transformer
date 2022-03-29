import torch
import pandas as pd
import hydra

class TowerDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        filename,
        date_channel,
        data_channels
    ):
        self.data = pd.read_csv(filename)
        self.data[date_channel] = pd.to_datetime(self.data[date_channel])
        self.data.set_index(date_channel)
        
        for data_channel in data_channels: assert data_channel in self.data, f"Channel {data_channel} is missing in {filename}"

@hydra.main(config_path=None, config_name="config")
def main(cfg):

    ds = TowerDataset(
        filename=cfg.dataset.filename,
        date_channel=cfg.dataset.channels.date,
        data_channels=cfg.dataset.channels.data
    )


if __name__  == "__main__":
    main()