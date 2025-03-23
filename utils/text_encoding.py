from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder

def categorical_encoding(text_list):
  label_encoder = LabelEncoder()
  label_encoder.fit(text_list)
  encoded_data = label_encoder.fit_transform(text_list)
  return encoded_data, label_encoder

def one_hot_encoding(text_list):
  text_reshaped = np.array(text_list).reshape(-1, 1)
  encoder = OneHotEncoder(sparse_output=False) 
  one_hot = encoder.fit_transform(text_reshaped)
  return one_hot, encoder