# Trial2Vec
[![PyPI version](https://badge.fury.io/py/transtab.svg)](https://badge.fury.io/py/trial2vec)
[![Downloads](https://pepy.tech/badge/trial2vec)](https://pepy.tech/project/trial2vec)
![GitHub Repo stars](https://img.shields.io/github/stars/ryanwangzf/trial2vec)
![GitHub Repo forks](https://img.shields.io/github/forks/ryanwangzf/trial2vec)

Wang, Zifeng and Sun, Jimeng. (2022). Trial2Vec: Zero-Shot Clinical Trial Document Similarity Search using Self-Supervision. Findings of EMNLP'22.

# News
- 12/8/2022: Support `download_embedding` that obtains the pretrained embedding only. It saves a lot of GPU/CPU memory! Please refer this [example](example/demo_download_embedding.ipynb) for detailed use cases.

```python
from trial2vec import download_embedding
t2v_emb = download_embedding()
```

- 10/27/2022: Support `word_vector` and `sentence_vector`!
```python
# sentence vectors
inputs = ['I am a sentence', 'I am another sentence']
outputs = model.sentence_vector(inputs)
# torch.tensor w/ shape [2, 128]
```

```python
# word vectors
inputs = ['I am a sentence', 'I am another sentence abcdefg xyz']
outputs = model.word_vector(inputs)
# {'word_embs': torch.tensor w/ shape [2, max_token, 128], 'mask': torch.tensor w/ shape [2, max_token]}
```


# Usage
Get pretrained Trial2Vec model in three lines:

```python
from trial2vec import Trial2Vec

model = Trial2Vec()

model.from_pretrained()
```

A jupyter example is shown at https://github.com/RyanWangZf/Trial2Vec/blob/main/example/demo_trial2vec.ipynb.

# How to install
Install the correct `PyTorch` version by referring to https://pytorch.org/get-started/locally/.

Then install `Trial2Vec` by

```bash
# Recommended because it is update to date, small bugs will be kept fixed
pip install git+https://github.com/RyanWangZf/Trial2Vec.git

```

or
```bash

pip install trial2vec

```

# Search similar trials
Use `Trial2Vec` to search similar clinical trials:

```python

# load demo data
from trial2vec import load_demo_data
data = load_demo_data()

# contains trial documents
test_data = {'x': data['x']} 

# make prediction
pred = model.predict(test_data)
```

# Encode trials

Use `Trial2Vec` to encode clinical trial documents:

```python

test_data = {'x': df} # contains trial documents

emb = model.encode(test_data) # make inference

# or just find the pre-encoded trial documents
emb = [model[nct_id] for test_data['x']['nct_id']]
```

# Continue training

One can continue to train the pretrained models on new trials as

```python

# just formulate trial documents as the format of `data`
data = load_demo_data()

model.fit(
    {
    'x':data['x'], # document dataframe
    'fields':data['fields'], # attribute field columns
    'ctx_fields':data['ctx_fields'], # context field columns
    'tag': data['tag'], # nct_id is the unique tag for each trial
    },
    valid_data={
            'x':data['x_val'],
            'y':data['y_val']
        },
)

# save
model.save_model('./finetuned-trial2vec')

```


