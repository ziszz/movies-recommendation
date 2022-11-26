import pandas as pd


def merge_dataset(data1_path: str, data2_path: str):
    df1 = pd.read_csv(data1_path)
    df2 = pd.read_csv(data2_path)
    
    df_final = df1.merge(df2)
    df_final.to_csv("data/merge/movie_rating.csv",index=False)
    return "Merge dataset success"