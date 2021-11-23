import time
import logging
import datetime
import os
import shutil
import torch
from easydict import EasyDict as edict
from torchvision import datasets, transforms


def _edict_to_dict(ed):
    result = {}
    for key, val in ed.items():
        if isinstance(val, edict):
            result[key] = _edict_to_dict(val)
        else:
            result[key] = val
    return result

# core utils

def create_exp_dir_name(exp_name, include_date=False):
    if include_date:
        # e.g. 'DCGAN-May-08-2020-5-30-12'
        suffix = '-' + datetime.datetime.now().strftime("%B-%d-%Y-%H-%M-%S")
    else:
        suffix = ''
    return f'{exp_name}{suffix}'


def create_save_model_directory(config, logger, dir_include_date=False):
    dir_name = create_exp_dir_name(config.exp_name, 
                                   include_date=dir_include_date)

    if dir_include_date:
        while os.path.exists(os.path.join(config.experiments_data_path, dir_name)):
            time.sleep(1.0)
            dir_name = create_exp_dir_name(config.exp_name, 
                                           include_date=dir_include_date)


    exp_path = os.path.join(config.experiments_data_path, dir_name)

    if not os.path.isdir(exp_path):
        folders = ['checkpoints', 'logs', 'tensorboard']
        for folder_name in folders:
            path = os.path.join(exp_path, folder_name)
            os.makedirs(path)

    log_path = os.path.join(exp_path, 'logs/log.txt')
    file_handler_abs_paths = [os.path.abspath(h.baseFilename)
                              for h in logger.handlers 
                              if isinstance(h, logging.FileHandler)]
    if os.path.abspath(log_path) not in file_handler_abs_paths:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.info(f'Experiment directory: {exp_path}')

    if not os.path.isdir(exp_path):
        if hasattr(config, 'config_path'):
            config_filename = os.path.basename(config.config_path)
            shutil.copyfile(config.config_path, os.path.join(exp_path, config_filename))
        else:
            with open(os.path.join(exp_path, 'config.yaml'), 'w') as file:
                documents = yaml.dump(_edict_to_dict(config), file, 
                                      default_flow_style=False)
        logger.info(f'Created experiment directory system')
    else:
        logger.info('Experiment directory already exists.')

    return exp_path

# agent utils
def dataset_dict_to_data_loader_dict(dataset_dict, batch_size, test_batch_size, use_cuda):
    result = {}
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if 'train' in dataset_dict:
        result['train'] = torch.utils.data.DataLoader(
            dataset_dict['train'], batch_size=batch_size, shuffle=True, **kwargs)
    if 'val' in dataset_dict:
        result['val'] = torch.utils.data.DataLoader(
            dataset_dict['val'], batch_size=test_batch_size, shuffle=True,
            **kwargs)
    if 'test' in dataset_dict:
        result['test'] = torch.utils.data.DataLoader(
            dataset_dict['test'], batch_size=test_batch_size, shuffle=True,
            **kwargs)

    return result

# trainer utils
# these follow the style of BasicClassificationTrainer

def create_training_state_dict(trainer):
    # Assumption: key matches attribute name
    missing_attrs = set(trainer.TRAINING_DICT_ATTRS) - set(vars(trainer))
    if len(missing_attrs) > 0:
        trainer.logger.warning(f'Missing training dict attributes: {missing_attrs}')
    return {attr: getattr(trainer, attr) for attr in trainer.TRAINING_DICT_ATTRS}


def _make_state_dict(trainer):
    state = create_training_state_dict(trainer)

    if hasattr(trainer, 'model'):
        state.update({'model': trainer.model.state_dict()})
    elif hasattr(trainer, 'models'):
        state['models'] = {}
        for model_name, model in trainer.models.items():
            state['models'][model_name] = model.state_dict()
    else:
        trainer.logger.critical('missing model in trainer class')

    if hasattr(trainer, 'optimizer'):
        state.update({'optimizer': trainer.optimizer.state_dict()})
    elif hasattr(trainer, 'optimizers'):
        state['optimizers'] = {}
        for opt_name, opt in trainer.optimizers.items():
            state['optimizers'][opt_name] = opt.state_dict()
    else:
        trainer.logger.critical('missing optimizer in trainer class')
    
    return state


def _load_state_dict(trainer, state):
    for attr in trainer.TRAINING_DICT_ATTRS:
        setattr(trainer, attr, state[attr])

    for attr in trainer.STATE_DICT_ATTRS:
        if hasattr(trainer, attr):
            if isinstance(getattr(trainer, attr), dict):
                for level_two_name, val in getattr(trainer, attr).items():
                    val.load_state_dict(state[attr][level_two_name])
            else:
                getattr(trainer, attr).load_state_dict(state.get(attr))


def load_checkpoint(trainer, checkpoint_dir, file_name="checkpoint.pth.tar"):
    path = os.path.join(checkpoint_dir, file_name)
    try:
        trainer.logger.info("Loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        _load_state_dict(trainer, checkpoint)
        trainer.logger.info(
            "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
            .format(checkpoint_dir, trainer.current_epoch, trainer.current_iter))
    except OSError as e:
        trainer.logger.info("No checkpoint exists from '{}'. Skipping...".format(checkpoint_dir))
        trainer.logger.info("**First time to train**")


def save_checkpoint(trainer, checkpoint_dir, file_name="checkpoint.pth.tar", is_best=0):
    state = _make_state_dict(trainer)
    checkpoint_path = os.path.join(checkpoint_dir, file_name)
    best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')

    # Save the state
    trainer.logger.info('Saving checkpoint: {}'.format(checkpoint_path))
    torch.save(state, checkpoint_path)
    # If it is the best copy it to another file 'model_best.pth.tar'
    if is_best:
        trainer.logger.info('Saving best checkpoint: {}'.format(checkpoint_path))
        shutil.copyfile(checkpoint_path, best_path)
