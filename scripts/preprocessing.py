import sys
import hydra
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from datetime import date
from datetime import datetime as dt
import yaml


def zscore(x, window=100):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


def fill_nan(df, fill_nan_limit, cols):
    for col in cols:
        if fill_nan_limit == 0:
            df[col] = df[col].interpolate(method='linear')
            df = df.ffill() # usefull if last elements are nan
            df = df.bfill() # usefull if first elements are nan
        else:
            df[col]=df[col].interpolate(limit=fill_nan_limit , limit_area='inside', method='linear')
    return df


@hydra.main(config_path=None, config_name="config")
def main(cfg):

    # load dataset    
    df = pd.read_csv(cfg.preprocessing.data.filename, sep=cfg.preprocessing.data.separator)
    
    # select rows by time
    df['Date time'] = pd.to_datetime(df['Date time'])
    time_mask = (df['Date time'] >= pd.Timestamp(cfg.preprocessing.start_date)) & (df['Date time'] <= pd.Timestamp(cfg.preprocessing.end_date))
    df = df[time_mask]
    df.set_index('Date time')
    df = df.iloc[:, :-1]

    # select channels (specifyed in the config file)
    thresholds = cfg.preprocessing.thresholds
    keys = list(thresholds.keys())
    df = df[df.columns.intersection(keys)] 

    # convert data type
    cols=[i for i in keys if i not in ["Date time"]] 
    for col in cols:
        df[col]=pd.to_numeric(df[col], errors='coerce')
    
    # remove outliers both by hardcoded threshold and z-score method
    for col in cols:
        df.loc[df[col] <= thresholds[col][0], col] = np.nan
        df.loc[df[col] >= thresholds[col][1], col] = np.nan
        df.loc[abs(zscore(df[col], 100)) > 3, col] = np.nan

    # fill nan element
    df = fill_nan(df, cfg.preprocessing.fill_nan_limit, cols)

    # time resampling
    df = df.resample('60min', on='Date time').mean()

    # fill nan element
    df = fill_nan(df, 4, cols)

    df['Date time'] = df.index

    # save processed data
    print(df.shape)
    df.to_csv(cfg.dataset.filename)

    start_date = df['Date time'].min()
    end_date = df['Date time'].max()

    for i in range(start_date.year, end_date.year):
        print('-----------------------', i, '-----------------------')
        time_mask = (df['Date time'].dt.year == i)
        df_tmp = df[time_mask] 
        print('{:<30s} {:<10s}'.format('CHANNEL', 'PERCENTAGE'))
        for col in cols:
            percentage_ = str( round( (len(df_tmp[col]) - len([x for x in np.isnan(df_tmp[col]) if x==True])) / len(df_tmp[col])* 100, 3 ) ) + '%'
            print('{:<30s} {:<10s}'.format(col, percentage_))

if __name__  == "__main__":
    main()

