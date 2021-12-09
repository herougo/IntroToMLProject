import os
import torch
from ml.utils.config import check_required_config_params
from ml.codebank.utils import (create_save_model_directory)
from ml.experiments.common.basic_trainer import BasicTrainer
from ml.experiments.common.memory_network_model import MemoryNetworkModel
from ml.experiments.common.babi_dataset import get_babi_dataloaders


class MemoryNetAgent:
    REQUIRED_CONFIG_PARAMS = ('exp_name experiments_data_path data_path test_batch_size '
                              'lr epochs batch_size max_n_batches tb_freq task_ids').split()

    def __init__(self, config, logger, dir_include_date=False):
        # boilerplate onwards
        self.config = config
        self.logger = logger
        self.config.use_cuda = torch.cuda.is_available() and not config.get('no_cuda', False)

        check_required_config_params(self.config, self.REQUIRED_CONFIG_PARAMS)
        self.exp_path = create_save_model_directory(config, logger, 
                                                    dir_include_date=dir_include_date)
        self.checkpoint_dir = os.path.join(self.exp_path, 'checkpoints')

        self.data_loaders, self.data_loader_metadata = None, None
        self.model = None
        self.optimizer = None
        self.callbacks = []
        self.trainer = None

    def setup(self):
        self.data_loaders, self.tensoriser, self.data_loader_metadata = self.get_data_loaders()
        self.model = self.get_model()
        if self.config.use_cuda:  # recommended: cuda before optimizer
            self.model = self.model.cuda()

        self.optimizer = self.get_optimizer()
        self.callbacks = []
        self.trainer = self.get_trainer()

    def get_model(self):
        sentence_len = self.data_loader_metadata.sentence_len
        vocab_size = self.data_loader_metadata.vocab_size
        n_sentence_line_types = self.data_loader_metadata.n_sentence_line_types
        n_question_line_types = self.data_loader_metadata.n_question_line_types
        word_map = self.tensoriser._wordmap
        return MemoryNetworkModel(sentence_len, vocab_size, n_sentence_line_types, n_question_line_types,
                                  self.config.use_cuda, word_map, **self.config.model_kwargs)

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def get_data_loaders(self):
        # return: e.g. {'train': train_loader, 'val': val_loader}
        data_loaders, tensoriser, data_loader_metadata = get_babi_dataloaders(self.config.task_ids,
                                                                              batch_size=self.config.batch_size)
        return data_loaders, tensoriser, data_loader_metadata

    def get_trainer(self):
        dataset_keys = ('sentence_sequence', 'question', 'sentence_line_type', 'question_line_type', 'story_lengths',
                        'label')
        return BasicTrainer(self.config, self.exp_path, self.model, self.optimizer,
                            self.logger, dataset_keys=dataset_keys, callbacks=self.callbacks)

    def train(self):
        try:
            self.trainer.train(data_loader=self.data_loaders['train'],
                               val_data_loader=self.data_loaders['val'])
        except KeyboardInterrupt as ex:
            self.logger.info('Keyboard interrupt. Exiting immediately...')

    def test(self):
        return self.trainer.test(data_loader=self.data_loaders['test'])