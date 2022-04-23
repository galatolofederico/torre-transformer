import sys
import hydra
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from datetime import date
from datetime import datetime as dt
import yaml



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
    print(df.shape)


    # select channels (specifyed in the config file)
    thresholds = cfg.preprocessing.thresholds
    keys = list(thresholds.keys())
    df = df[df.columns.intersection(keys)] 


    # convert data type
    cols=[i for i in keys if i not in ["Date time"]] 
    for col in cols:
        df[col]=pd.to_numeric(df[col], errors='coerce')
    

    # remove outliers 
    for col in cols:
        df.loc[df[col] <= thresholds[col][0], col] = np.nan
        df.loc[df[col] >= thresholds[col][1], col] = np.nan


    # fill nan element
    for col in cols:
        df[col] = df[col].interpolate(method='nearest')
        df = df.ffill() # usefull if last elements are nan
        df = df.bfill() # usefull if first elements are nan


    # time resampling
    df = df.resample('60min', on='Date time').mean()

    # fill nan
    for col in cols:
        df[col] = df[col].interpolate(method='nearest')

    df['Date time'] = df.index

    # save processed data
    print(df.shape)
    df.to_csv(cfg.dataset.filename)

if __name__  == "__main__":
    main()

