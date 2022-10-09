# Trial2Vec
Findings of EMNLP'22 | Trial2Vec: Zero-Shot Clinical Trial Document Similarity Search using Self-Supervision

# Useage
Get pretrained Trial2Vec model in three lines:

```python
from trial2vec import Trial2Vec

model = Trial2Vec()

model.from_pretrained()
```

# How to install
Install the correct `PyTorch` version by referring to https://pytorch.org/get-started/locally/.

Then install `Trial2Vec` by

```bash

pip install git+https://github.com/RyanWangZf/Trial2Vec.git

```

# Search similar trials
Use `Trial2Vec` to search similar clinical trials:

```python

test_data = {'x': df} # contains trial documents

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

