import os
import glob
import pdb
import subprocess
import sys

def check_model_dir(experiment_id, root_dir='./experiments_records'):
    """
    Check whether the checkouts/results folders of current experiment(exp_id) exist,
        If not, will create both folders
    Parameters
    ----------
    experiment_id : str, optional (default='init.test')
        name of current experiment

    root_dir :  str,
        root dir of current project
    """
    if os.path.exists(root_dir) is False:
        os.mkdir(root_dir)
    exp_root = os.path.join(root_dir, experiment_id)
    if os.path.exists(exp_root) is False:
        os.mkdir(exp_root)
    checkout_dir = os.path.join(exp_root, 'checkpoints')
    result_dir = os.path.join(exp_root, 'results')
    if os.path.exists(checkout_dir) is False:
        os.mkdir(checkout_dir)
    if os.path.exists(result_dir) is False:
        os.mkdir(result_dir)

def check_checkpoint_file(input_dir, suffix='pth.tar'):
    '''
    Check whether the `input_path` is directory or to the checkpoint file.
        If it is a directory, find the only 'pth.tar' file under it.

    Parameters
    ----------
    input_path: str
        The input path to the pretrained model.

    suffix: 'pth.tar' or 'model'
        The checkpoint file suffix;
        If 'pth.tar', the saved model is a torch model.
        If 'model', the saved model is a scikit-learn based model.
    '''
    suffix = '.' + suffix
    if input_dir.endswith(suffix):
        return input_dir

    ckpt_list = glob.glob(os.path.join(input_dir, '*'+suffix))
    assert len(ckpt_list) <= 1, f'Find more than one checkpoints under the dir {input_dir}, please specify the one to load.'
    assert len(ckpt_list) > 0, f'Do not find any checkpoint under the dir {input_dir}.'
    return ckpt_list[0]

def check_model_config_file(input_dir):
    '''
    Check whether the `input_path` is directory or to the `model_config.json` file.
        If it is a directory, find the only '.json' file under it.

    Parameters
    ----------
    input_path: str
        The input path to the pretrained model.

    '''
    if input_dir.endswith('.json'):
        return input_dir

    if not os.path.isdir(input_dir):
        # if the input_dir is the given checkpoint model path,
        # we need to find the config file under the same dir.
        input_dir = os.path.dirname(input_dir)

    ckpt_list = glob.glob(os.path.join(input_dir, '*.json'))

    if len(ckpt_list) == 0:
        return None

    # find model_config.json under this input_dir
    model_config_name = [config for config in ckpt_list if 'model_config.json' in config]
    if len(model_config_name) == 1:
        return model_config_name[0]

    # if no model_config.json found, retrieve the only .json file.
    assert len(ckpt_list) <= 1, f'Find more than one config .json under the dir {input_dir}.'
    return ckpt_list[0]

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])