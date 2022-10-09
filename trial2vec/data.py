'''
Implement dataset for trial search model training.
'''
import pdb

import pandas as pd
from transformers import AutoTokenizer
from torch import Tensor, device

class TrialDataCollator:
    '''The basic trial data collator.
    Subclass it and override the `__init__` & `__call__` function if need operations inside this step.
    Returns
    -------
    batch_df: pd.DataFrame
        A dataframe contains multiple fields for each trial.
    '''
    def __init__(self) -> None:
        # subclass to add tokenizer
        # subclass to add feature preprocessor
        pass

    def __call__(self, examples):
        batch_df = pd.concat(examples, 0)
        batch_df.fillna('none',inplace=True)
        return batch_df
        
class TrialSearchCollator(TrialDataCollator):
    '''
    The basic collator for trial search tasks.
    '''
    def __init__(self,
        bert_name,
        max_seq_length,
        ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.max_length = max_seq_length

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return batch

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            if 'cuda' in target_device:
                batch[key] = batch[key].cuda()
    return batch