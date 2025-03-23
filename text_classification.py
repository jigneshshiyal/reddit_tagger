import pandas as pd
from model_script.train_model import train_model_text_cls

df = pd.read_csv('./data/process_data.csv')

result = train_model_text_cls(df["text"].tolist(), df["label"].tolist())
print(result)