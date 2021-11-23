import time
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import os
from ml.metrics.meter_metrics import AverageMeter
from ml.utils.tensorboard_wrapper import TensorboardWrapper
from ml.utils.training_utils import EarlyStopping, create_input_dict
from ml.utils.config import check_required_config_params
from ml.codebank.utils import (save_checkpoint, load_checkpoint)

class BasicTrainer:
    # TRAINING_DICT_ATTRS and STATE_DICT_ATTRS are used by helper functions for
    # determining attribute access for checkpointing
    TRAINING_DICT_ATTRS = 'current_epoch total_iter'.split()
    STATE_DICT_ATTRS = 'model models optimizer optimizers'.split()
    REQUIRED_CONFIG_PARAMS = 'epochs max_n_batches use_cuda tb_freq'.split()

    def __init__(self, config, exp_path, model, optimizer, logger, callbacks=None):
        self.config = config  # easydict
        self.exp_path = exp_path
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.callbacks = callbacks or []

        # boilerplate onwards
        check_required_config_params(self.config, self.REQUIRED_CONFIG_PARAMS)
        self.checkpoint_dir = os.path.join(self.exp_path, 'checkpoints')
        self.tb_writer = TensorboardWrapper(os.path.join(self.exp_path, 'tensorboard'))

        self.current_epoch = 0
        self.total_iter = 0
        self.current_iter = 0
        self.load_checkpoint()

    def load_checkpoint(self, file_name="checkpoint.pth.tar"):
        load_checkpoint(self, self.checkpoint_dir, file_name)

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        save_checkpoint(self, self.checkpoint_dir, file_name, is_best)

    def train(self, data_loader, val_data_loader=None, patience=None):
        early_stopping = EarlyStopping(patience=patience)

        # on_train_begin #################################
        train_loss = AverageMeter()
        ################################# on_train_begin #

        while self.current_epoch < self.config.epochs:
            # use prefetch_generator and tqdm for iterating through data
            pbar = tqdm(enumerate(BackgroundGenerator(data_loader)),
                        total=len(data_loader))
            start_time = time.time()

            # for loop going through dataset
            for i, data in pbar:
                if i >= self.config.max_n_batches:
                    break

                self.current_iter = i
                self.model.train()
                val_loss = None
                input_dict = create_input_dict(data, self.config.use_cuda)

                # It's very good practice to keep track of preparation time and computation time using
                # tqdm to find any issues in your dataloader
                prepare_time = start_time - time.time()

                ##################################################
                # train_step #####################################
                output_dict = self.model(input_dict)
                train_batch_loss = output_dict['loss']

                self.optimizer.zero_grad()
                train_batch_loss.backward()
                self.optimizer.step()
                ##################################### train_step #
                ##################################################

                # compute computation time and *compute_efficiency*
                process_time = start_time - time.time() - prepare_time
                pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                    process_time / (process_time + prepare_time),
                    self.current_epoch, self.config.epochs))
                start_time = time.time()

                self.total_iter += 1
                
                ##################################################
                # on_batch_end ###################################
                
                # Make sure you use .item() to detach from the computation graph so 
                # the graph doesn't get saved.
                train_loss.update(train_batch_loss.item(), input_dict['x'].size()[0])

                if (self.total_iter - 1) % self.config.tb_freq == 0:
                    val_loss = self.validate(val_data_loader)

                    # udpate tensorboard
                    self.tb_writer.add_scalar('model/train_loss', train_batch_loss.item(),
                                              self.total_iter)
                    if val_data_loader is not None:
                        self.tb_writer.add_scalar('model/val_loss', val_loss, self.total_iter)
                ################################### on_batch_end #
                ##################################################
            
            ##################################################
            # on_epoch_end ###################################
            self.current_epoch += 1
            val_loss = self.validate(val_data_loader)
            stop, is_best = early_stopping.report_loss(val_loss)
            metrics = self.model.get_metrics(reset=True)
            print_msg = ', '.join([
                f'Train loss: {train_loss.val}', f'Val loss: {val_loss}',
            ] + [f'{metric_name}: {metric}' for metric_name, metric in metrics.items()])
            self.logger.info(print_msg)
            for metric_name, metric in metrics.items():
                self.tb_writer.add_scalar(f'model/{metric_name}', metric, self.total_iter)
            self.save_checkpoint(is_best=is_best)
            ################################### on_epoch_end #
            ##################################################

            if stop:
                self.logger.info('Stopping Early')
                break

    def validate(self, val_data_loader):
        if val_data_loader is None:
            return None

        training = self.model.training
        self.model.eval()

        # use prefetch_generator and tqdm for iterating through data
        pbar = enumerate(BackgroundGenerator(val_data_loader))

        val_loss = AverageMeter()

        # for loop going through dataset
        for i, data in pbar:
            if i >= self.config.max_n_batches:
                break

            input_dict = create_input_dict(data, self.config.use_cuda)
            output_dict = self.model(input_dict, skip_metrics=True)
            batch_val_loss = output_dict['loss']
            val_loss.update(batch_val_loss.item(), input_dict['x'].size()[0])

        if training:
            self.model.train()

        return val_loss.value

    def test(self, data_loader):
        if data_loader is None:
            return

        training = self.model.training
        self.model.eval()

        # use prefetch_generator and tqdm for iterating through data
        pbar = enumerate(BackgroundGenerator(data_loader))

        test_loss = AverageMeter()
        self.model.get_metrics(reset=True)

        # for loop going through dataset
        for i, data in pbar:
            if i >= self.config.max_n_batches:
                break

            input_dict = create_input_dict(data, self.config.use_cuda)
            output_dict = self.model(input_dict)
            test_loss.update(output_dict['loss'].item(), x.size()[0])

        if training:
            self.model.train()

        return {'test_loss': test_loss.val, **self.model.get_metrics(reset=True)}
