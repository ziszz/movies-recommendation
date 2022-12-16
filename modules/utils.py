import os

import pandas as pd
from absl import logging


def merge_dataset(data1_path: str, data2_path: str):
    try:
        df1 = pd.read_csv(data1_path)
        df2 = pd.read_csv(data2_path)
        
        df_final = df1.merge(df2)
        
        if not os.path.exists("data/merge"):
            os.makedirs("data/merge")
            
        df_final.to_csv("data/merge/movie_rating.csv",index=False)
        
        return "Merge dataset success"
    except BaseException as err:
        logging.error(f"ERROR IN merge_dataset:\n{err}")