import sys
import hydra
import pandas as pd
from datetime import datetime
from datetime import timedelta
import yaml

@hydra.main(config_path=None, config_name="config")
def main(cfg):
    df = pd.read_csv(cfg.dataset.filename)
    df[cfg.dataset.channels.date] = pd.to_datetime(df[cfg.dataset.channels.date])
    start_date = datetime.strptime(cfg.args.start_date, "%d/%m/%Y")
    end_date = datetime.strptime(cfg.args.end_date, "%d/%m/%Y")
    total_days = (end_date - start_date).days
    days_per_split = total_days // cfg.args.time_splits
    
    print(f"start_date: {start_date}")
    print(f"end_date: {end_date}")
    print(f"total_days: {total_days}")
    print(f"days_per_split: {days_per_split}")

    splits = dict(
        train=dict(dates=[]),
        validation=dict(dates=[]),
        test=dict(dates=[]),
    )

    current_start_date = start_date
    for time_split_i in range(0, cfg.args.time_splits):
        splits_days = dict(
            train = int(days_per_split*cfg.args.distribution.train),
            validation = int(days_per_split*cfg.args.distribution.validation),
            test = int(days_per_split*cfg.args.distribution.test)
        )
        for split in splits.keys():
            current_end_date = current_start_date + timedelta(days=splits_days[split]-1)
            splits[split]["dates"].append({
                "from": current_start_date.strftime("%d/%m/%Y"),
                "to": current_end_date.strftime("%d/%m/%Y")
            })

            current_start_date = current_end_date + timedelta(days=1)

    print(yaml.dump(splits))

if __name__  == "__main__":
    sys.argv.append("args=compute-splits")
    main()