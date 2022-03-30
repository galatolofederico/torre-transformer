import hydra
import matplotlib.pyplot as plt

from src.dataset import get_dataset


@hydra.main(config_path=None, config_name="config")
def main(cfg):
    ds = get_dataset(cfg, "train")
    subplots = (3, 3)
    figsize = (10, 10)

    for elem in ds:
        fig, axes = plt.subplots(*subplots, figsize=figsize)

        for channel, channel_name, ax in zip(elem.T, ds.channel_names, axes.flatten()):
            ax.set_title(channel_name)
            ax.plot(channel)
        
        plt.show()
        plt.close(fig)

if __name__  == "__main__":
    main()