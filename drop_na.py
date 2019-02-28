import pandas as pd

def drop_na(file):
    df = pd.read_csv(file)
    df = df.dropna(how='any', axis=0)
    df.to_csv(file)
    return True

file = r'.\data\pku_foreign_tags.csv'
drop_na(file)