import sys
import hydra
import numpy as np
from tqdm import tqdm
import json

from src.dataset import get_dataset

@hydra.main(config_path=None, config_name="config")
def main(cfg):
    ds = get_dataset(cfg, "train")
    samples = dict()

    for elem, sample in tqdm(zip(ds, range(0, cfg.args.samples)), total=cfg.args.samples):
        for channel, channel_name in zip(elem.T, ds.channel_names):
            if channel_name not in samples: samples[channel_name] = np.zeros((cfg.args.samples, cfg.dataset.window))
            samples[channel_name][sample, :] = channel 

    stats = dict()
    for channel_name in ds.channel_names:
        channel_samples = samples[channel_name].flatten()
        channel_samples = channel_samples[~np.isnan(channel_samples)]
        stats[channel_name] = dict(
            channel=channel_name,
            mean=channel_samples.mean(),
            std=channel_samples.std()
        )
    
    print(json.dumps(stats, indent=4))

if __name__  == "__main__":
    sys.argv.append("args=dataset-stats")
    main()