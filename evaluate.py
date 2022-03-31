import argparse
import os
from torch.utils.data import DataLoader
import hydra
from hydra.utils import get_original_cwd

from src.dataset import get_dataset
from src.model import TransformerRegressor

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    model_path = os.path.join(get_original_cwd(), cfg.evaluate.model)
    assert os.path.exists(model_path), "you must specify a model with evaluate.model=<path-to-model>"
    ds = get_dataset(cfg, cfg.evaluate.dataset)
    dl = DataLoader(
        ds,
        batch_size=cfg.evaluate.batch_size,
        num_workers=os.cpu_count()
    )

    model = TransformerRegressor.load_from_checkpoint(model_path)
    
    
if __name__  == "__main__":
    main()