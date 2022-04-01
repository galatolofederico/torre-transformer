import sys
import hydra
import numpy as np
from tqdm import tqdm
import yaml

from src.dataset import get_dataset

@hydra.main(config_path=None, config_name="config")
def main(cfg):
    ds = get_dataset(cfg, cfg.args.split, use_stats=cfg.args.use_stats)
    samples = dict()

    for elem, sample in tqdm(zip(ds, range(0, cfg.args.samples)), total=cfg.args.samples):
        for channel, channel_name in zip(elem.T, ds.channel_names):
            if channel_name not in samples: samples[channel_name] = np.zeros((cfg.args.samples, cfg.dataset.window))
            samples[channel_name][sample, :] = channel 

    stats = list()
    for channel_name in ds.channel_names:
        channel_samples = samples[channel_name].flatten()
        channel_samples = channel_samples[~np.isnan(channel_samples)]
        stats.append(dict(
            channel=channel_name,
            mean=float(channel_samples.mean()),
            std=float(channel_samples.std())
        ))
    
    print(cfg.args.split)
    print(yaml.dump(stats))

if __name__  == "__main__":
    sys.argv.append("args=dataset-stats")
    main()