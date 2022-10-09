'''
Provide a general trainer functions for most deep learning
based methods.
'''
import math
import pdb
import os
import json
from pickletools import optimize
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections.abc import Mapping
from collections import defaultdict
import warnings

import numpy as np
import torch
from torch.cuda.amp import autocast
import transformers
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from .check import make_dir_if_not_exist
from .parallel import DataParallelModel, unwrap_model

#TODO: accelerator (https://github.com/huggingface/accelerate) deployed for multi-gpu training.
class Trainer:
    '''
    A general trainer used to train deep learning models.

    Parameters
    ----------
    model: nn.Module
        The model to be trained.

    train_objectives: list[tuple[DataLoader, nn.Module]]
        The defined pairs of dataloaders and the loss models.

    test_data: (optional) dict or Dataset
        Depending on the implemented `get_test_dataloader` function.
        That function receives it as inputs and return test dataloader.

    test_metric: (optional) str
        Which test metric is used to judge the best checkpoint during the training.
        Only used when `test_data` is given. Should be contained in
        the returned metric dict by `evaluate` function.

    less_is_better: (optional) bool
        If the test metric is less the better.
        Ignored if no test_data and test_metric is given.

    load_best_at_end: bool
        If load the best checkpoint at the end of training.

    n_gpus: int
        How many GPUs used to kick of training.
        If set larger than 1, parallel training will be used.

    output_dir: str
        The intermediate model checkpoints during the training
        will be dump to under this dir.

    Examples
    --------
    >>> trainer = Trainer(
    ... model=model,
    ... train_objectives=[(dataloader1, loss_model1), (dataloader2, loss_model2)],
    ... )
    >>> trainer.train(
    ... epochs=10,
    ... )
    '''
    evaluated=False
    def __init__(self,
        model: nn.Module,
        train_objectives: List[Tuple[DataLoader, nn.Module]],
        test_data = None,
        test_metric = None,
        less_is_better = False,
        load_best_at_end = True,
        n_gpus = 1,
        output_dir = './checkpoints/',
        **kwargs,
        ):
        self.train_dataloader = [dataloader for dataloader,_, in train_objectives]
        self.loss_models = [loss_model for _,loss_model in train_objectives]
        self.model = model
        if test_data is not None: self.test_dataloader = self.get_test_dataloader(test_data)
        else: self.test_dataloader = None
        self.test_metric = test_metric
        self.less_is_better = less_is_better
        self.load_best_at_end = load_best_at_end
        self.output_dir = output_dir
        make_dir_if_not_exist(output_dir)
        if self.test_dataloader is not None:
            self.best_dir = os.path.join(self.output_dir, 'best')
            make_dir_if_not_exist(self.best_dir)

        # TODO add multi gpu support
        self.n_gpus = n_gpus
        if self.n_gpus > 1:
            from pytrial.utils.parallel import DataParallelModel, DataParallelCriterion
            self.model = self._wrap_model(self.model, device_ids=range(self.n_gpus))

    def train(self,
        epochs=10,
        learning_rate=2e-5,
        weight_decay=1e-4,
        warmup_ratio=0,
        scheduler='warmupcosine',
        evaluation_steps=10,
        max_grad_norm=0.5,
        use_amp=False,
        **kwargs,
        ):
        '''
        Kick of training using the provided loss model and train dataloaders.

        Parameters
        ----------
        epochs: int, optional (default=10)
            Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.

        learning_rate: float, optional (default=3e-5)
            The learning rate.

        weight_decay: float, optional (default=1e-4)
            Weight decay applied for regularization.

        warmup_ratio: float (default=0)
            How many steps used for warmup training. If set 0, not warmup.

        scheduler: str (default='warmupcosine')
            Pick learning rate scheduler for warmup. Ignored if warmup_ratio <= 0.
            in ['constantlr','warmupconstant','warmuplinear','warmupcosine','warmupcosinewithhardrestarts'].

        evaluation_steps: int (default=10)
            How many iterations while we print the training loss and
            conduct evaluation if evaluator is given.

        max_grad_norm: float (default=0.5)
            Clip the gradient to avoid NaN in some cases.

        use_amp: bool (default=False)
            Whether or not use mixed precision training.
        '''
        self.best_score = np.inf if self.less_is_better else -np.inf

        self.score_logs = defaultdict(list)
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        num_examples = self._compute_num_train_samples(self.train_dataloader[0])
        steps_per_epoch, num_train_steps = self._build_data_iterator(epochs=epochs)
        warmup_steps = self._compute_warmup_steps(num_train_steps=num_train_steps, warmup_ratio=warmup_ratio)
        
        optimizer_param = {'lr':learning_rate, 'weight_decay':weight_decay}
        self.optimizers, self.schedulers = self._build_optimizer(scheduler=scheduler,
            loss_models=self.loss_models,
            warmup_steps=warmup_steps,
            num_train_steps=num_train_steps,
            optimizer_param=optimizer_param,
            weight_decay=weight_decay,
            )
        
        print("***** Running training *****")
        print(f"  Num examples = {num_examples}")
        print(f"  Num Epochs = {epochs}")
        print(f"  Total optimization steps = {num_train_steps}")

        train_loss_dict = defaultdict(list)
        global_step = 0
        for epoch in tqdm(range(epochs), desc='Training Epoch'):
            training_steps = 0
            self.model.train()

            for _ in tqdm(range(steps_per_epoch), desc='Iteration'):

                # train models in one iter
                self.train_one_iteration(
                        max_grad_norm=max_grad_norm,
                        warmup_steps=warmup_steps,
                        use_amp=use_amp, 
                        scaler=scaler,
                        train_loss_dict=train_loss_dict
                    )

                training_steps += 1
                global_step += 1

                if evaluation_steps>0 and global_step % evaluation_steps == 0:
                    
                    self.model.eval()
                    
                    self.evaluated = True

                    print('\n######### Train Loss {} #########'.format(global_step))
                    for key in train_loss_dict.keys():
                        print('{} {:.4f} \n'.format(key, np.mean(train_loss_dict[key])))

                    if scheduler is not None and warmup_steps > 0:
                        for scheduler_ in self.schedulers:
                            print('learning rate: {:.6f}'.format(scheduler_.get_lr()[0]))
                    
                    train_loss_dict = defaultdict(list)
                    if self.test_dataloader is not None:
                        metrics = self.evaluate()

                        print('\n######### Eval {} #########'.format(global_step))
                        for metric, metric_value in metrics.items():
                            self.score_logs[metric].append(metric_value)
                            print('{}: {:.4f}'.format(metric, metric_value))

                        if self.test_metric in metrics:
                            metric_value = metrics[self.test_metric]

                        elif self.test_metric is None:
                            warnings.warn(f'The `test_metric` was not specified, use the default metric `{metric}` as the test metric for judging early stopping.')

                        else:
                            raise ValueError(f'Do not find the specified metric `{self.test_metric}` from the evaluation results, please check either `test_metric` or the `trainer.evalute` function.')
                        
                        # update best score and save checkpoints
                        best_score = self._update_best_metric(self.best_score, metric_value, self.less_is_better)
                        if best_score != self.best_score:
                            self.score_logs['best_global_step'] = global_step
                            model = unwrap_model(self.model)
                            model.save_model(self.best_dir)
                            self.best_score = best_score

                            best_score = self._update_best_metric(self.best_score, metric_value, self.less_is_better)
                            if best_score != self.best_score:
                                self.score_logs['best_global_step'] = global_step
                                model = unwrap_model(self.model)
                                model.save_model(self.best_dir)
                                self.best_score = best_score

                    # save model checkpoint
                    output_dir = os.path.join(self.output_dir, f'./{global_step}')
                    make_dir_if_not_exist(output_dir)
                    model = unwrap_model(self.model)
                    model.save_model(output_dir)

        # after training
        self._save_log(self.output_dir)
        if self.load_best_at_end and self.test_dataloader is not None and self.evaluated:
            print(f'Load best ckpt from `{self.best_dir}`.')
            self.model.load_model(self.best_dir)

        print('Training completes.')

    def evaluate(self):
        '''
        Need to be created by specific tasks.

        Returns
        -------
        A dict of computed evalution metrics.
        '''
        raise NotImplementedError

    def get_test_dataloader(self, test_data):
        '''
        Need to be created by specific tasks.
        '''
        raise NotImplementedError

    def prepare_input(self, data):
        '''
        Need to be reimplemented sometimes when input data is not in the standard dict structure.
        '''
        if len(data) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model."
            )

        if isinstance(data, Mapping):
            return type(data)({k: self.prepare_input(v) for k, v in data.items()})

        elif isinstance(data, (tuple,list)):
            return type(data)(self.prepare_input(v) for v in data)

        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.model.device)
            return data.to(**kwargs)

        return data

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

            # get a batch of data from the target data_iterator
            data = self._get_a_train_batch(data_iterator=data_iterator, train_idx=train_idx)
            data = self.prepare_input(data)

            # update model by backpropagation
            if use_amp:
                loss_value, skip_scheduler, scale_before_step = self._update_one_iteration_amp(loss_model=loss_model, data=data, optimizer=optimizer, scaler=scaler, max_grad_norm=max_grad_norm)
            else:
                loss_value = self._update_one_iteration(loss_model=loss_model, data=data, optimizer=optimizer, max_grad_norm=max_grad_norm)

            train_loss_dict[train_idx].append(loss_value.item())

            if use_amp:
                skip_scheduler = scaler.get_scale() != scale_before_step

            if not skip_scheduler and warmup_steps > 0:
                scheduler.step()

    def _build_data_iterator(self, epochs):
        self.data_iterators = [iter(dataloader) for dataloader in self.train_dataloader]
        steps_per_epoch = max([len(dataloader) for dataloader in self.train_dataloader])
        num_train_steps = int(steps_per_epoch * epochs)
        return  steps_per_epoch, num_train_steps

    def _build_optimizer(self,
        loss_models,
        warmup_steps,
        num_train_steps,
        optimizer_param,
        weight_decay,
        scheduler=None,
        ):
        optimizer_class = torch.optim.AdamW
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            print(optimizer_param)
            param_optimizer = list(loss_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_param)
            if warmup_steps > 0:
                scheduler_obj = self._build_scheduler(optimizer, scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)
            else:
                scheduler_obj = None
            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)
        return optimizers, schedulers

    def _build_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _update_best_metric(self, best_score, metric_value, less_is_better):
        if less_is_better:
            if metric_value < best_score:
                return metric_value
            else:
                return best_score
        else:
            if metric_value > best_score:
                return metric_value
            else:
                return best_score

    def _save_log(self, output_dir):
        log_filename = os.path.join(output_dir, 'train_logs.json')
        with open(log_filename, 'w') as f:
            f.write(json.dumps(self.score_logs))

    def _wrap_model(self, model):
        # wrap model to be paralleled
        model = DataParallelModel(model)
        return model

    def _compute_num_train_samples(self, dataloader):
        if isinstance(dataloader, DataLoader):
            return len(dataloader.dataset)
        else:
            # if there are more than one dataloaders for a loss_model,
            # iterate to find the last dataloader item
            return self._compute_num_train_samples(dataloader[0])

    def _compute_warmup_steps(self, num_train_steps, warmup_ratio):
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) if warmup_ratio > 0 else 0
        return warmup_steps


    def _update_one_iteration(self, loss_model, data, optimizer, max_grad_norm):
        loss_model_return = loss_model(data)
        loss_value = loss_model_return['loss_value']
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
        optimizer.step()
        return loss_value

    def _update_one_iteration_amp(self, loss_model, data, optimizer, scaler, max_grad_norm):
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
        return loss_value, skip_scheduler, scale_before_step

    def _get_a_train_batch(self, data_iterator, train_idx):
        try:
            data = next(data_iterator)

        except StopIteration:
            data_iterator = iter(self.train_dataloader[train_idx])
            self.data_iterators[train_idx] = data_iterator
            data = next(data_iterator)
            
        return data