'''
Implement Trial2Vec model for clinical trial similarity search.
'''

import pdb
import os
import math
from collections import defaultdict
import random


from sklearn.neighbors import NearestNeighbors
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from .trainer import Trainer
from .check import install_package, make_dir_if_not_exist
from .check import check_checkpoint_file, check_model_dir, check_model_config_file
from .base import TrialSearchBase
from .losses import InfoNCELoss
from .data import TrialSearchCollator, batch_to_device
from .metrics import precision, recall, ndcg

# may be changed later
# PRETRAINED_TRIAL2VEC_URL = 'https://uofi.box.com/shared/static/wmvi33l3ne4hl88lui0h0v2l2j75smyb.zip'
PRETRAINED_TRIAL2VEC_URL = 'https://storage.googleapis.com/pytrial/trial2vec_pretrained.zip'

class BuildModel(nn.Module):
    config = {}
    def __init__(self,
        model_name,
        emb_dim,
        fields=None,
        ctx_fields=None,
        device=None,
        ) -> None:
        super().__init__()
        self.device = device
        self.base_encoder = AutoModel.from_pretrained(model_name)
        self.global_proj_head = nn.Linear(768, emb_dim, bias=False)
        self.local_proj_head = nn.Linear(768, emb_dim, bias=False)
        self.multihead_att = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=6,
            batch_first=False, # [seq, batch, feature]
        )
        self.config['fields'] = fields
        self.config['ctx_fields'] = ctx_fields

    def forward(self, inputs, return_local_emb=True):
        '''
        (1) Go through each attr and ctx, encode them into embds.
        (2) Go through multihead att over the component embds.
        (3) Apply local and global proj head to get each component embds
        and global trial embds.

        Parameters
        ----------
        inputs: dict[dict]
            A dict of input components after tokenized.
            E.g., inputs['title'] are the tokenized title texts,
            w/ keys like input_ids, attention_mask, etc.

        Returns
        -------
        local_embs: dict[Tensor]
            A dict of all component embds.

        global_embs: Tensor
            The trial-level embeds.
        '''
        local_embs = {}

        fields = self.config['fields']
        ctx_fields = self.config['ctx_fields']

        attr_embs = self._encode_fields(fields, inputs, local_embs) # num_attr, bs, emb_dim

        ctx_embs = self._encode_fields(ctx_fields, inputs, local_embs) # num_ctx, bs, emb_dim

        if len(ctx_embs) > 0:
            # take a fusion of context and attribute embeddings
            temp_embs, att_score = self.multihead_att(ctx_embs, attr_embs, attr_embs) # num_ctx, bs, emb_dim
            temp_embs = torch.mean(temp_embs, 0)
        else:
            # no context embeddings, take directly the average of all local embs
            temp_embs = attr_embs.mean(0) # bs, emb_dim

        # projection and normalize
        global_embs = self.global_proj_head(temp_embs)
        # global_embs = global_embs / global_embs.norm(dim=-1, keepdim=True)

        return_dict = {
            'global_embs':global_embs,
        }

        if return_local_emb:
            return_dict['local_embs'] = local_embs

        return return_dict

    def _encode_fields(self, fields, inputs, local_embs):
        emb_list = []
        for fd in fields:
            input_dict = {}
            for k, v in inputs.items():
                if fd in k:
                    tempkey = k.replace(fd+'_','')
                    input_dict[tempkey] = v
            if len(input_dict) == 0:
                continue
            input_dict = batch_to_device(input_dict, self.device)
            res = self.base_encoder(**input_dict, return_dict=True)
            emb_list.append(res['pooler_output'])
            temp_emb = self.local_proj_head(res['pooler_output'])

            # normalize local embeddings
            # temp_emb = temp_emb / temp_emb.norm(dim=-1, keepdim=True)
            local_embs[fd] = temp_emb
        
        if len(emb_list) > 0:
            embs = torch.stack(emb_list)
            return embs
        else:
            return emb_list
    
    def _encode_text(self, inputs, projection):
        # encode tokenized texts into embeddings
        inputs = batch_to_device(inputs, self.device)
        res = self.base_encoder(**inputs, return_dict=True)
        if projection:
            temp_emb = self.local_proj_head(res['pooler_output'])
        else:
            temp_emb = res['pooler_output']
        return temp_emb
    
    def _encode_word(self, inputs):
        # encode tokenized words into embeddings
        inputs = batch_to_device(inputs, self.device)
        res = self.base_encoder(**inputs, return_dict=True)
        temp_emb = self.local_proj_head(res['last_hidden_state'])
        return temp_emb

class LocalMatchCollator(TrialSearchCollator):
    def __init__(self,
        bert_name,
        max_seq_length,
        fields,
        ctx_fields,
        tag_field=None,
        is_train=True,
        device='cuda:0',
        ) -> None:
        super().__init__(
            bert_name=bert_name,
            max_seq_length=max_seq_length,
        )
        self.fields = fields
        self.ctx_fields = ctx_fields
        self.tag_field = tag_field
        self.is_train = is_train
        self.device = device
        if is_train:
            print('Trigger training for Trial2Vec, will load `nltk` and `textaugment`.')
            install_package('nltk')
            install_package('textaugment')
            import nltk
            from textaugment import EDA
            nltk.download('stopwords')
            nltk.download('omw-1.4')
            nltk.download('wordnet')
            self.eda = EDA()

    def __call__(self, features):
        return_dict = defaultdict(list)
        batch_df = pd.DataFrame(features)
        batch_df.fillna('', inplace=True)

        fields = self.fields
        ctx_fields = self.ctx_fields

        if self.is_train:
            fields = self._random_sample(fields)
            ctx_fields = self._random_sample(ctx_fields)

        return_dict.update(self._batch_tokenize(batch_df=batch_df, fields=fields))
        return_dict.update(self._batch_tokenize(batch_df=batch_df, fields=ctx_fields))

        if self.tag_field is not None:
            return_dict[self.tag_field] = batch_df[self.tag_field].tolist()
        return return_dict

    def _batch_tokenize(self, batch_df, fields):
        return_dict = {}
        for field in fields:

            if self.is_train:
                texts = self._eda_augment(batch_df[field])
            else:
                texts = batch_df[field].tolist()

            tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            return_dict[field] = tokenized
        return return_dict

    def _eda_augment(self, text):
        eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
        new_text = text.apply(lambda x: eda_aug(x) if len(x)>1 else x)
        new_text = new_text.apply(lambda x: x[0] if isinstance(x,list) else x).tolist()
        return new_text

    def _random_sample(self, fields, n=2):
        fields = np.array(fields).copy()
        if random.random() > 0.5:
            num_select = np.random.poisson(n, 1)[0]
            num_select = np.maximum(num_select, 1)
            sub_fields = fields[:num_select].tolist()
            return sub_fields
        else:
            return fields

class LocalMatchLoss(InfoNCELoss):
    '''
    Match local attribute embeddings and trial-level global embeddings.
    '''
    def __init__(self, model, fields, logit_scale_init_value):
        super().__init__(model=model, logit_scale_init_value=logit_scale_init_value)
        self.fields = fields

    def forward(self, inputs):
        outputs = self.model(inputs)
        embs = outputs['global_embs']
        fields = [f for f in self.fields if f in outputs['local_embs']]
        field = np.random.choice(fields,1)[0]
        local_embs = outputs['local_embs'][field]
        logits_per_trial = self.compute_logits(embs, local_embs)
        logits_per_attr = logits_per_trial.t()
        loss = (self.compute_loss(logits_per_trial) + self.compute_loss(logits_per_attr)) / 2
        return {'loss_value':loss}

class GlobalMatchCollator(LocalMatchCollator):
    def __init__(self,
        bert_name,
        max_seq_length,
        fields,
        ctx_fields,
        tag_field=None,
        is_train=True,
        device='cuda:0',
        ) -> None:
        super().__init__(
            bert_name=bert_name,
            max_seq_length=max_seq_length,
            fields=fields,
            ctx_fields=ctx_fields,
            tag_field=tag_field,
            is_train=is_train,
            device=device,
        )

    def __call__(self, features):
        return_dict = dict()
        batch_df = pd.DataFrame(features)
        batch_df.fillna('', inplace=True)

        fields = self.fields
        ctx_fields = self.ctx_fields

        # process to get anchor
        if self.is_train:
            fields = self._random_sample(fields)
            ctx_fields = self._random_sample(ctx_fields)
            
        if random.random() > 0.5: 
            ctx_fields = [] # only take local embeddings
            ctx_inputs = {}
        else:
            ctx_inputs = self._batch_tokenize(batch_df=batch_df, fields=ctx_fields)

        # anchor is attr+ctx
        return_dict['anchor'] = self._batch_tokenize(batch_df=batch_df, fields=fields)
        return_dict['anchor'].update(ctx_inputs)

        # positive is attr+ctx
        return_dict['pos'] = self._batch_tokenize(batch_df=batch_df, fields=fields)
        return_dict['pos'].update(ctx_inputs)

        # negative is attr'+ctx
        # process to get negative by random shuffling attr field texts
        sub_fields = self._random_sample(fields)
        batch_df[sub_fields] = batch_df[sub_fields].sample(frac=1).reset_index(drop=True)
        return_dict['neg'] = self._batch_tokenize(batch_df=batch_df, fields=fields)
        return_dict['neg'].update(ctx_inputs)

        if self.tag_field is not None:
            return_dict[self.tag_field] = batch_df[self.tag_field].tolist()
        return return_dict

class GlobalMatchLoss(InfoNCELoss):
    '''
    Match positive trials (by replacing trial components).
    '''
    def __init__(self, model, fields, logit_scale_init_value):
        super().__init__(model=model, logit_scale_init_value=logit_scale_init_value)
        self.fields = fields

    def forward(self, inputs):
        outputs_pos = self.model(inputs['pos'], return_local_emb=False)
        outputs_neg = self.model(inputs['neg'], return_local_emb=False)
        outputs_anc = self.model(inputs['anchor'], return_local_emb=False)
        temp_emb = torch.cat([outputs_pos['global_embs'], outputs_neg['global_embs']], dim=0)
        logits_per_trial = self.compute_logits(outputs_anc['global_embs'], temp_emb)
        loss = self.compute_loss(logits_per_trial)
        return {'loss_value': loss}

class Trial2VecTrainer(Trainer):
    '''
    Subclass the original trainer and provide specific evaluation functions.
    '''
    def get_test_dataloader(self, test_data):
        self.test_dataloader = test_data
        return self.test_dataloader

    def prepare_input(self, inputs):
        return self.model._prepare_input(inputs)

    def train_one_iteration(self, 
        max_grad_norm=None,
        warmup_steps=None,
        use_amp=None, 
        scaler=None,
        train_loss_dict=None):
        '''
        Default training one iteration steps, can be subclass can reimplemented.
        '''
        skip_scheduler = False
        num_train_objectives = len(self.train_dataloader)
        for train_idx in range(num_train_objectives):
            data_iterator = self.data_iterators[train_idx]
            loss_model = self.loss_models[train_idx]
            loss_model.zero_grad()
            loss_model.train()
            optimizer = self.optimizers[train_idx]
            scheduler = self.schedulers[train_idx]
            # print(scheduler.get_lr())

            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader[train_idx])
                self.data_iterators[train_idx] = data_iterator
                data = next(data_iterator)

            # usually pass raw tensors to the target device
            if 'anchor' in data:
                # global match loss
                new_data = {}
                for k,v in data.items():
                    new_data[k] = self.prepare_input(v)
                data = new_data
            else:
                # local match loss
                data = self.prepare_input(data)

            if use_amp:
                with autocast():
                    loss_return = loss_model(data)
                loss_value = loss_return['loss_value']
                scale_before_step = scaler.get_scale()
                scaler.scale(loss_value).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                skip_scheduler = scaler.get_scale() != scale_before_step
            else:
                loss_model_return = loss_model(data)
                loss_value = loss_model_return['loss_value']
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                optimizer.step()

            train_loss_dict[train_idx].append(loss_value.item())

            if use_amp:
                skip_scheduler = scaler.get_scale() != scale_before_step
                if not skip_scheduler:
                    scheduler.step()
            else:
                if warmup_steps > 0:
                    scheduler.step()

    def evaluate(self):
        # encode all trials in test data
        val_doc_embs = self.model.encode(self.model.val_doc_inputs, return_dict=True)
        self.model.update_emb(val_doc_embs)
        return_dict = self.model.evaluate(test_data={'x':self.test_dataloader['x'],'y':self.test_dataloader['y'],})
        return return_dict

class Trial2Vec(TrialSearchBase):
    '''
    Implement the Trial2Vec model for trial document similarity search.

    References: Wang, Z., & Sun, J. (2022). Trial2Vec: Zero-Shot Clinical Trial Document Similarity Search using Self-Supervision. Findings of EMNLP 2022.

    Parameters
    ----------
    fields: list[str]
        A list of fields of documents used as the `attribute` fields by Trial2Vec model.

    ctx_fields: list[str]
        A list of fields of documents used as the `context` fields by Trial2Vec model.

    tag_field: str
        The tag indicating trial documents, default to be 'nct_id'.

    bert_name: str (default='emilyalsentzer/Bio_ClinicalBERT')
        The base transformer-based encoder. Please find model names
        from the model hub of transformers (https://huggingface.co/models).

    emb_dim: int, optional (default=768)
        Dimensionality of the embedding vectors.

    logit_scale_init_value: float, optional (default=0.07)
        The logit scale or the temperature.

    max_seq_length: int (default=128)
        The maximum length of input tokens for the base encoder.

    epochs: int, optional (default=10)
        Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.

    batch_size: int, optional (default=64)
        Number of samples in each training batch.

    learning_rate: float, optional (default=3e-5)
        The learning rate.

    weight_decay: float, optional (default=1e-4)
        Weight decay applied for regularization.

    warmup_ratio: float (default=0)
        How many steps used for warmup training. If set 0, not warmup.

    evaluation_steps: int (default=10)
        How many iterations while we print the training loss and
        conduct evaluation if evaluator is given.

    num_workers: int, optional (default=0)
        Use these many worker threads to train the model (=faster training with multicore machines).

    device: str or torch.device (default='cuda:0')
        The device to put the model on.

    use_amp: bool (default=False)
        Whether or not use mixed precision training.

    experiment_id: str, optional (default='test')
        The name of current experiment.
    '''
    trial_embs = {}
    val_doc_inputs = None

    def __init__(self,
        fields=None,
        ctx_fields=None,
        tag_field='nct_id',
        bert_name='emilyalsentzer/Bio_ClinicalBERT',
        emb_dim=128,
        logit_scale_init_value=0.07,
        max_seq_length=128,
        epochs=10,
        batch_size=64,
        learning_rate=2e-5,
        weight_decay=1e-4,
        warmup_ratio=0,
        evaluation_steps=10,
        num_workers=0,
        device='cuda:0',
        use_amp=False,
        experiment_id='test'
        ) -> None:
        super().__init__(experiment_id=experiment_id)
        self.config = {
            'max_seq_length':max_seq_length,
            'logit_scale_init_value':logit_scale_init_value,
            'bert_name':bert_name,
            'emb_dim':emb_dim,
            'epochs':epochs,
            'batch_size':batch_size,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
            'evaluation_steps':evaluation_steps,
            'num_workers':num_workers,
            'device':device,
            'use_amp':use_amp,
            'warmup_ratio':warmup_ratio,
            'tag_field':tag_field,
            'fields':fields,
            'ctx_fields':ctx_fields,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        if num_workers > 1:
            # disable tokenizer parallel when doing data collate parallel
            os.environ['TOKENIZERS_PARALLELISM'] = '0'
        self.use_amp = use_amp
        self.device = device if isinstance(device, str) else 'cuda:0'

        self._build_model()

    def fit(self, train_data, valid_data=None):
        '''Train the trial2vec model to get document embeddings for trial search.

        Parameters
        ----------
        train_data: {
                    'x': pd.DataFrame,
                    'fields': list[str],
                    'ctx_fields': list[str],
                    'tag': str,
                    }
            Training corpus for the model. `x`: a dataframe of trial documents.
            `fields`: optional, the fields of documents to use for training as key attributes. If not given,
            the model uses all fields in `x`.
            `ctx_fields`: optional, the fields of documents which belong to context components. If not given,
            the model will only learn from `fields`.
            `tag`: optional, the field in `x` that serves as unique identifiers. Typically it
            is the `nct_id` of each trial. If not given, the model takes integer tags.

        valid_data: {'x':pd.DataFrame 'y':np.ndarray}.
            Validation data used for identifying the best checkpoint during the training.
            Need to rewrite the function:
            `get_val_dataloader`.

        '''
        self._input_data_check(train_data)
        self._fit(train_data, val_data=valid_data)

    def encode(self,
        inputs,
        batch_size=None,
        num_workers=None,
        return_dict=True,
        verbose=True,
        ):
        '''
        Encode input documents and output the document embeddings.

        Parameters
        ----------
        inputs: {
                'x': pd.DataFrame,
                'fields': list[str],
                'ctx_fields': list[str],
                'tag': str,
                }
                Share the same input format as the `train_data` in
                `fit` function.
                If `fields`, `ctx_fields`, `tag` are not given,
                will reuse the ones used during training.

        batch_size: int, optional
            The batch size when encoding trials.
        
        num_workers: int, optional
            The number of workers when building the val dataloader.

        return_dict: bool
            If set True, return dict[np.ndarray].
            Else, return np.ndarray with the order same as the input documents.

        verbose: bool
            Whether plot progress bar or not.

        Returns
        -------
        embs: dict[np.ndarray]
            Encoded trial-level embeddings with key (tag) and value (embedding)..
        '''
        self._input_data_check(inputs)

        if batch_size is not None:
            self.config['batch_size'] = batch_size
        
        if num_workers is not None:
            self.config['num_worker'] = num_workers

        # build dataset and dataloader
        dataloader = self.get_val_dataloader(inputs)

        # i = iter(dataloader)
        # res = next(i)

        # go through dataloader and encode trial embds
        emb_list = []
        tag_list = []

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, desc='Encoding', disable=not verbose):
                if self.config['tag_field'] is not None:
                    tag = data.pop(self.config['tag_field'])
                    tag_list.append(tag)

                data = self._prepare_input(data)
                outputs = self.model(data)

                temp_emb = outputs['global_embs'].cpu()
                temp_emb = temp_emb / temp_emb.norm(dim=-1, keepdim=True)
                temp_emb = temp_emb.numpy()

                if len(temp_emb.shape) == 1: temp_emb = temp_emb[None]
                emb_list.append(temp_emb)

        embs = np.concatenate(emb_list, 0)

        if len(tag_list) == 0:
            tag_list = list(range(len(embs)))
        else:
            tag_list = sum(tag_list, [])

        if return_dict:
            embs = dict(zip(tag_list, embs))
            return embs
        else:
            return (tag_list, embs)

    def sentence_vector(self, inputs, no_grad=True, projection=True):
        '''
        Encode input sentence or list of sentences.

        Parameters
        ----------
        inputs: str or List[str]
            Input sentence(s).
        
        no_grad: bool
            Whether to use torch.no_grad() or not. If set False, the Trial2Vec model can be updated using the output embeddings.
        
        projection: bool
            Whether to project the sentence embeddings to the same space as the trial embeddings.
            Before projection, dim=768. After projection, dim=128.
        
        Returns
        -------
        embs: np
        '''
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt', max_length=self.config['max_seq_length'])
        if no_grad:
            with torch.no_grad():
                outputs = self.model._encode_text(inputs, projection=projection)
        else:
            outputs = self.model._encode_text(inputs, projection=projection)
        return outputs
    
    def word_vector(self, inputs, no_grad=True):
        '''
        Encode input sentence or list of sentences into word embeddings.
        
        Parameters
        ----------
        inputs: str or List[str]
            A sentence that you want to get word-level embeddings. The model will return a **dict** of word-level embeddings and 
            the corresponding mask.
        
        no_grad: bool
            Whether to use torch.no_grad() or not. If set False, the Trial2Vec model can be updated using the output embeddings.
            
        Returns
        -------
        outputs: dict
            A dict of word-level embeddings and the corresponding mask.
        '''
        if isinstance(inputs, str):
            inputs = [inputs]

        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt', max_length=self.config['max_seq_length'])
        if no_grad:
            with torch.no_grad():
                outputs = self.model._encode_word(inputs)
        else:
            outputs = self.model._encode_word(inputs)
        
        outputs = {'word_embs': outputs, 'mask': inputs['attention_mask']}
        return outputs

    def predict(self,
        test_data,
        top_k=10,
        return_df=True):
        '''
        Predict the top-k relevant for input documents.

        Parameters
        ----------
        test_data: {
                'x': pd.DataFrame,
                'fields': list[str],
                'ctx_fields': list[str],
                'tag': str,
                }
                Share the same input format as the `train_data` in
                `fit` function.
                If `fields`, `ctx_fields`, `tag` are not given,
                will reuse the ones used during training.

        top_k: int
            Number of retrieved candidates.

        return_df: float
            If return dataframe for the computed similarity ranking.
            If set True, return (rank, sim);
            else, return rank_list=[[(doc1,sim1),(doc2,sim2)], [(doc1,sim1),...]].

        Returns
        -------
        rank: pd.DataFrame
            A dataframe contains the top ranked NCT ids for each.

        sim: pd.DataFrame
            A dataframe contains the corresponding similarities.

        rank_list: list[list[tuple]]
            A list of tuples of top ranked docs and similarities.
        '''
        self._input_data_check(test_data)
        
        tag_field = self.config['tag_field']

        all_embs = np.stack(self.trial_embs.values())
        all_tags = np.stack(self.trial_embs.keys())

        # TODO: skip encoding those already stored in model
        # to_encode_test_trial = test_trial[~test_trial[tag_field].isin(all_tags)]
        tags, embs = self.encode(test_data, return_dict=False)
        
        # rank for each test document
        sim = embs.dot(all_embs.T)
        rank = np.argsort(sim, 1)[:,::-1] # flip
        rank = rank[:,1:top_k+1]

        if return_df:
            tag_list, sim_list = [], []
            for i,row in enumerate(rank):
                tag_list.append(all_tags[row])
                sim_list.append(sim[i][row])
            tag_ar = np.array(tag_list)
            sim_ar = np.array(sim_list)
            rank_df = pd.DataFrame(tag_ar, columns=[f'rank_{i}' for i in range(top_k)], index=tags)
            sim_df = pd.DataFrame(sim_ar, columns=[f'rank_{i}' for i in range(top_k)], index=tags)
            return rank_df, sim_df
        else:
            rank_list = []
            for i,row in enumerate(rank):
                tag_ar = all_tags[row]
                sim_ar = sim[i][row]
                rank_list.append(
                    list(zip(tag_ar, sim_ar))
                )
            return rank_list

    def evaluate(self, test_data):
        '''
        Evaluate within the given trial and corresponding candidate trials.

        x =
        | target_trial | trial1 | trial2 | trial3 |
        | nct01        | nct02  | nct03  | nct04  |

        y =
        | label1 | label2 | label3 |
        | 0      | 0      | 1      |

        Parameters
        ----------
        test_data: {
            'x': pd.DataFrame,
            'y': pd.DataFrame
            }
            The provided labeled dataset for test trials. Follow the format listed above.

        Returns
        -------
        results: dict[float]
            A dict of metrics and the values.

        '''
        test_df = test_data['x']
        label_df = test_data['y']
        ranked_label_list = []
        for idx, row in test_df.iterrows():
            target_trial = row['target_trial']
            if target_trial in self.trial_embs:
                target_emb = self.__getitem__(target_trial)
            else:
                raise ValueError(f'The embeddings of trial {target_trial} are not found.')
            candidate_embs = np.stack([self.__getitem__(tag) for tag in row.values[1:]])
            sim = target_emb[None].dot(candidate_embs.T)[0]
            labels = label_df.iloc[idx].to_numpy()
            if labels.sum() == 0: continue
            ranked_label = labels[np.argsort(sim)[::-1]]
            ranked_label_list.append(ranked_label)
        ranked_label_list = np.array(ranked_label_list)

        return_dict = {}
        for k in [1,2,5]:
            return_dict[f'precision@{k}'] = precision(ranked_label_list, k)
            return_dict[f'recall@{k}'] = recall(ranked_label_list, k)
        return_dict[f'ndcg@{k}'] = ndcg(ranked_label_list, k)
        return return_dict

    def load_model(self, checkpoint):
        '''
        Load model and the pre-encoded trial embeddings from the given
        checkpoint dir.

        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.
            If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            If a filepath, will load from this file.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        state_dict = torch.load(checkpoint_filename, map_location=torch.device('cpu'))
        if config_filename is not None:
            config = self._load_model_config(config_filename)
            self.config.update(config)
            self.model.config.update({'fields':config['fields'], 'ctx_fields':config['ctx_fields']})
        self.model.load_state_dict(state_dict['model'])
        self.trial_embs = state_dict['emb']

    def save_model(self, output_dir):
        make_dir_if_not_exist(output_dir)
        self._save_model_config(model_config=self.config, output_dir=output_dir)
        model = self._unwrap_model(self.model)
        self._save_checkpoint(
            {'model':model.state_dict(),'emb':self.trial_embs},
            output_dir=output_dir)

    def get_train_dataloader(self, inputs):
        df = inputs['x']
        dataset_list = self._build_dataset(df=df)
        collator_list = self._build_collator()
        dataloader_list = self._build_dataloader(dataset_list, collator_list)
        return dataloader_list

    def get_val_dataloader(self, inputs):
        df = inputs['x']
        dataset = self._build_val_dataset(df=df)
        collate_fn = self._build_val_collator()
        dataloader = self._build_val_dataloader(dataset=dataset, collate_fn=collate_fn)
        return dataloader

    def update_emb(self, emb_dict):
        '''
        Update trial embeds: add or modify.

        Parameters
        ----------
        emb_dict: dict[np.ndarray]
            The tag and corresponding trial embeddings to updated.
        '''
        self.trial_embs.update(
            emb_dict
        )
    
    def from_pretrained(self, input_dir=None):
        '''
        Download pretrained Trial2Vec model.
        '''
        if input_dir is None:
            input_dir = './trial_search/pretrained_trial2vec'

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            print(f'Download pretrained Trial2Vec model, save to {input_dir}.')
            self._download_pretrained(output_dir=input_dir)
        
        print('Load pretrained Trial2Vec model from', input_dir)
        self.load_model(input_dir)

    def _prepare_input(self, inputs):
        ctx_fields = self.config['ctx_fields']
        fields = self.config['fields']
        output_dict = {}
        for fd in ctx_fields:
            if fd not in inputs:
                continue
            for k,v in inputs[fd].items():
                output_dict[f'{fd}_{k}'] = v

        for fd in fields:
            if fd not in inputs:
                continue
            for k,v in inputs[fd].items():
                output_dict[f'{fd}_{k}'] = v
        return output_dict

    def __getitem__(self, tag):
        return self.trial_embs[tag]

    def _build_model(self):
        model = BuildModel(
            model_name=self.config['bert_name'],
            emb_dim=self.config['emb_dim'],
            ctx_fields=self.config['ctx_fields'],
            fields=self.config['fields'],
            device=self.device,
        )
        self.model = self._wrap_model(model, self.config['device'])

    def _fit(self, train_data, val_data=None):
        # build dataset
        if 'fields' in train_data: self.config['fields']= train_data['fields']
        if 'ctx_fields' in train_data: self.config['ctx_fields'] = train_data['ctx_fields']
        if 'tag' in train_data: self.config['tag_field'] = train_data['tag']
        self.model.config.update({'fields': self.config['fields'], 'ctx_fields':self.config['ctx_fields']})

        if val_data is not None:
            self._build_val_docs(train_data, val_data=val_data)

        loss_models = self._build_loss_model()

        # build train dataloader list
        dataloader_list = self.get_train_dataloader(train_data)

        # kick off training
        train_objectives = list(zip(dataloader_list, loss_models))
        trainer = Trial2VecTrainer(
            model=self,
            train_objectives=train_objectives,
            test_data=val_data,
            test_metric='ndcg@5',
        )

        trainer.train(
            **self.config,
        )

        # encode all training trial docs after training
        self.update_emb(self.encode(train_data, return_dict=True))

    def _build_collator(self, is_train=True):
        collator_list = [
            LocalMatchCollator(
                bert_name=self.config['bert_name'],
                max_seq_length=self.config['max_seq_length'],
                fields=self.config['fields'],
                ctx_fields=self.config['ctx_fields'],
                device=self.config['device'],
                is_train=is_train,
                tag_field=None,
            ),
            GlobalMatchCollator(
                bert_name=self.config['bert_name'],
                max_seq_length=self.config['max_seq_length'],
                fields=self.config['fields'],
                ctx_fields=self.config['ctx_fields'],
                device=self.config['device'],
                is_train=is_train,
                tag_field=None,
            ),
            ]
        return collator_list

    def _build_dataloader(self, dataset_list, collator_list):
        '''Build dataloaders for multiple training
        supervision of Trial2Vec.
        (1) query to the whole trial (w/ the query) search
        (2) replace query w/ others
        '''
        dataloader_list = []
        for i,dataset in enumerate(dataset_list):
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                collate_fn=collator_list[i],
                pin_memory=True,
            )
            dataloader_list.append(dataloader)
        return dataloader_list

    def _build_dataset(self, df):
        dataset_list = []

        # build local contrastive (match indiv component to global trial embs)
        dataset = Dataset.from_pandas(df)
        dataset.set_transform(self._dataset_transform)
        dataset_list.append(dataset)

        # build global contrastive (discriminate global trial emb when replacing attrs)
        dataset = Dataset.from_pandas(df)
        dataset.set_transform(self._dataset_transform)
        dataset_list.append(dataset)

        return dataset_list

    def _dataset_transform(self, examples):
        return_dict = {}

        if self.config['fields'] is not None:
            for k in self.config['fields']: 
                return_dict[k] = examples[k]
        
        if self.config['ctx_fields'] is not None:
            for k in self.config['ctx_fields']: 
                return_dict[k] = examples[k]

        if self.config['tag_field'] is not None:
            return_dict[self.config['tag_field'] ] = examples[self.config['tag_field'] ]

        if len(return_dict) == 0:
            return_dict = examples

        return return_dict

    def _build_loss_model(self):
        '''
        Build two loss models for training trial2vec from scratch.
        '''
        loss_models = [
            LocalMatchLoss(model=self.model, fields=self.config['fields'], logit_scale_init_value=self.config['logit_scale_init_value']),
            GlobalMatchLoss(model=self.model, fields=self.config['fields'], logit_scale_init_value=self.config['logit_scale_init_value']),
        ]
        return loss_models

    def _val_dataset_transform(self, examples):
        return self._dataset_transform(examples=examples)

    def _build_val_dataset(self, df):


        dataset = Dataset.from_pandas(df)
        dataset.set_transform(self._val_dataset_transform)
        return dataset

    def _build_val_dataloader(self, dataset, collate_fn):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return dataloader

    def _build_val_collator(self):
        return LocalMatchCollator(
            bert_name=self.config['bert_name'],
            max_seq_length=self.config['max_seq_length'],
            fields=self.config['fields'],
            ctx_fields=self.config['ctx_fields'],
            device=self.config['device'],
            is_train=False,
            tag_field=self.config['tag_field'],
        )

    def _build_val_docs(self, train_data, val_data):
        # extract trial docs from val data target trials and candidate trials
        df_tr = train_data['x']
        df_va = val_data['x']
        tag_field = self.config['tag_field']
        tag_field = tag_field if tag_field is not None else 'tag'
        df_va = pd.DataFrame({tag_field: df_va.to_numpy().flatten()}).drop_duplicates().reset_index(drop=True)
        df_va = df_va.merge(df_tr, on=tag_field, how='inner')
        self.val_doc_inputs = {
            'x':df_va,
        }

    def _input_data_check(self, inputs):
        '''
        Check the training / testing data fits the formats.
        Target to (1) check if inputs valid,
                    if not, give tips about the data problem.

        Parameters
        ----------
        inputs: {
                'x': pd.DataFrame,
                'fields': list[str],
                'ctx_fields': list[str],
                'tag': str,
                }
        '''
        # check overall input format
        assert 'x' in inputs, 'No input trial doc dataframe found in inputs.'
        df = inputs['x']
        if 'fields' in inputs:
            try:
                _ = df[inputs['fields']]
            except:
                raise Exception('Cannot find the specified `fields` in inputs dataframe.')
        if 'ctx_fields' in inputs:
            try:
                _ = df[inputs['ctx_fields']]
            except:
                raise Exception('Cannot find the specified `ctx_fields` in inputs dataframe.')
        if 'tag' in inputs:
            try:
                _ = df[inputs['tag']]
            except:
                raise Exception('Cannot find the specified `tag` in inputs dataframe.')

        # check data type
        try:
            _ = df.applymap(str)
        except:
            raise Exception('Cannot transform the input dataframe to str type, please check the inputs.')

    def _wrap_model(self, model, device):
        if isinstance(device, list):
            model = nn.DataParallel(model, device_ids=device)
            model.to(f'cuda:{model.device_ids[0]}')
        elif device == 'cpu':
            model.to(torch.device('cpu'))
        else:
            model.to(torch.device('cuda'))
        return model

    def _unwrap_model(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        else:
            return model

    def _download_pretrained(self, output_dir):
        import wget
        import zipfile
        filename = wget.download(url=PRETRAINED_TRIAL2VEC_URL, out=output_dir)
        zipf = zipfile.ZipFile(filename, 'r')
        zipf.extractall(output_dir)
        zipf.close()

class Trial2VecEmbedding:
    def __init__(self, trial_embs) -> None:
        self.trial_embs = trial_embs
        self.knn = None

    def build_knn(self, n_neighbors=10):
        '''
        Build a KNN model for the embeddings.

        Parameters
        ----------
        n_neighbors: int
            The number of neighbors to be returned.
        '''
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.knn.fit(np.array(list(self.trial_embs.values())))

    def search_topk(self, tag, k=10):
        '''
        Find similar trials using KNN model.

        Parameters
        ----------
        tag: list[str] or str
            The tag to be looked up in the model.
        
        k: int
            The number of neighbors to be returned.
        '''
        if self.knn is None:
            print('knn model is not built, building now...')
            self.build_knn(n_neighbors=k)
        
        if isinstance(tag, str):
            tag = [tag]

        for t in tag:
            assert t in self.trial_embs, f'The trial {t} is not found in the pretrained embeddings!'
        
        trial_embs = np.array([self.trial_embs[t] for t in tag])
        pred = self.knn.kneighbors(trial_embs, return_distance=False)
        pred_tags = []
        for p in pred:
            pred_tags.append([list(self.trial_embs.keys())[i] for i in p])
        pred_tags = np.array(pred_tags)
        if len(pred_tags) == 1:
            pred_tags = pred_tags[0]
        return pred_tags

    def keys(self):
        return self.trial_embs.keys()

    def __len__(self):
        return len(self.trial_embs)

    def __getitem__(self, tag):
        '''
        Get the embeddings of documents by the trial tags.

        Parameters
        ----------
        tag: str, int, list[str], list[int]
            The tag (or tags) to be looked up in the model.

        Returns
        -------
        The embeddings of each document.
        '''
        if isinstance(tag, str):
            tag = [tag]
        
        embs = []
        for t in tag:
            assert t in self.trial_embs, f'The trial {t} is not found in the pretrained embeddings!'
            embs.append(self.trial_embs[t])
        
        if len(embs) == 1:
            return embs[0]
        else:
            return np.stack(embs)

def download_embedding():
    '''
    Download the pretrained Trial2Vec model and only keep the embeddings and the indexer.

    Returns
    -------
    trial2vec_embs: Trial2VecEmbedding
        The embeddings of the pretrained Trial2Vec model.
        Can be used to retrieve similar trials directly
        or to initialize a new model for further training.
    '''
    import copy
    model = Trial2Vec(device='cpu')
    model.from_pretrained()
    trial_embs = copy.deepcopy(model.trial_embs)
    # remove the cache occupied by the model
    del model
    if torch.cuda.is_available():
        # clean up GPU memory
        torch.cuda.empty_cache()
    trial2vec_embs = Trial2VecEmbedding(trial_embs)
    return trial2vec_embs