'''
TODO: add .from_pretrained to load pretrained trial search model
from cloud storage.
'''
import abc
import os
import json
import pdb

import torch
import pandas as pd

from .check import check_model_dir
from .check import make_dir_if_not_exist

class TrialSearchBase(abc.ABC):
    '''Abstract class for all trial search algroithms.

    Parameters
    ----------
    experiment_id: str, optional (default = 'test')
        The name of current experiment.

    '''

    @abc.abstractmethod
    def __init__(self, experiment_id='test'):
        check_model_dir(experiment_id)
        self.checkout_dir = os.path.join('./experiments_records', experiment_id,
                                         'checkpoints')
        self.result_dir = os.path.join('./experiments_records', experiment_id,
                                       'results')
        make_dir_if_not_exist(self.checkout_dir)
        make_dir_if_not_exist(self.result_dir)


    @abc.abstractmethod
    def fit(self, train_data, valid_data):
        '''
        Parameters
        ----------

        train_data: {
                    'x': pd.DataFrame,
                    'fields': list[str],
                    'y': pd.Series or np.array,
                    }

            The input training sample dict.

        valid_data: {
                    'x',
                    'fields': list[str],
                    'y': pd.Series or np.array,
                    }

            The input validation sample dict.


        Returns
        -------

        self: object
            The trained model.

        '''
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, test_data):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, checkpoint):
        '''
        Parameters
        ----------

        checkpoint: str
            The path to the saved model.

        Returns
        -------

        self: object
            The loaded pretrained model.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, output_dir):
        '''
        Parameters
        ----------
        output_dir: str
            The directory to save the model states.

        '''
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, inputs):
        '''
        Encode input documents into embeddings.

        Parameters
        ----------
        inputs: dict
            The input documents.
        '''
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        self.model.train()
        return self
    
    def eval(self, mode=False):
        self.training = mode
        self.model.eval()
        return self

    @abc.abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError



    def _save_checkpoint(self, state,
                        epoch_id=0,
                        is_best=False,
                        output_dir=None,
                        filename='checkpoint.pth.tar'):
        if output_dir is None:
            output_dir = self.checkout_dir

        if epoch_id < 1:
            filepath = os.path.join(output_dir, 'latest.' + filename)
        elif is_best:
            filepath = os.path.join(output_dir, 'best.' + filename)
        else:
            filepath = os.path.join(self.checkout_dir,
                                    str(epoch_id) + '.' + filename)
        torch.save(state, filepath)

    def _save_model_config(self, model_config, output_dir=None):
        if output_dir is None:
            output_dir = self.checkout_dir
        temp_path = os.path.join(output_dir, "model_config.json")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with open(temp_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(model_config, indent=4))

    def _load_model_config(self, checkpoint=''):
        if checkpoint == '':
            temp_path = os.path.join(self.checkout_dir,
                                     'model_config.json')
            assert os.path.exists(
                temp_path), 'cannot find predictor_config.json, please it in dir {0}'.format(
                self.checkout_dir)
        else:
            temp_path = checkpoint
            assert os.path.exists(
                temp_path), 'cannot find checkpoint file from path: {0}'.format(
                checkpoint)
        print('load predictor config file from {0}'.format(temp_path))
        with open(temp_path, 'r') as f:
            predictor_config = json.load(f)
        return predictor_config

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

    def _process_dataframe(self, df, fields):
        if fields is not None:
            df = df[fields]
        if 'nct_id' in df:
            df = df.drop(['nct_id'], axis=1)
        df = df.applymap(str)
        df = df.apply(lambda x: x.name + ': ' + x)
        df = df.applymap(lambda x: x.lower())
        df_raw_texts = df.agg(' '.join, axis=1)
        df_raw_texts = pd.DataFrame(df_raw_texts, columns=['text'])
        return df_raw_texts