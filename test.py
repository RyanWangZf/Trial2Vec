import os
import pandas as pd

from trial2vec import Trial2Vec
from trial2vec import load_demo_data

model = Trial2Vec()
model.from_pretrained('./trial_search/pretrained_trial2vec')
emb = model['NCT01327170']
print(emb)
print(emb.shape)

data = load_demo_data('./demo_data')
print(data)

res = model.predict(data, return_df=True)
print(res)