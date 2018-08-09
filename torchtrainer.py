import os.path

import torch
import torch.nn as nn
import torch.utils.data

from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Callback(object):
    """Abstract base class used to build new callbacks.

    Note:
        The `metrics` dictionary that callback methods take as an argument will contain keys
        for all of the trainer metrics.
        For batch callbacks values correspond to evaluation results over batch.
        For epoch callbacks values are averaged over whole epoch.
        If validation dataset/split was passed, epoch callbacks will have metrics results
        for validation data with 'val_' prefix in keys e.g. 'val_loss'.
        Epoch and batch callbacks also take corresponding iteration number i.e.
        `epoch` takes current epoch number and `batch` takes current batch number.
        You can access Trainer in the Callback via `self.trainer`.
        `is_aborted` flag is `True` if training was aborted by some exception.
    """

    def on_train_begin(self):
        pass

    def on_train_end(self, is_aborted):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch, metrics):
        pass


class EarlyStopping(Callback):
    """Stops training if given metrics doesn't improve."""

    def __init__(self, metric='val_loss', patience=5, verbose=0):
        """Initialize EarlyStopping callback.

        Args:
            metrics (str): Metric name to keep track of. (Default: 'val_loss')
            patience (int): After how many epochs without improvement training should stop.
                (Default: 5)
            verbose (int): Two levels of verbosity:
                * 1 - show update information,
                * 0 - show nothing. (<- Default)
        """

        self.metric = metric
        self.patience = patience
        self.verbose = verbose

        self._best_value = float('inf')
        self._last_epoch = 0

    def on_epoch_end(self, epoch, metrics):
        if not self.metric in metrics:
            raise ValueError("There is no such metric evaluated: {}".format(self.metric))

        new_value = metrics[self.metric]
        if new_value < self._best_value:
            if self.verbose:
                print("Metric '{}' improved from {} to {}".format(
                    self.metric, self._best_value, new_value))
            self._best_value = new_value
            self._last_epoch = epoch
        elif self.verbose:
            print("Metric '{}' did not improved. Current best: {}".format(
                self.metric, self._best_value))

        if self._last_epoch + self.patience >= epoch:
            if self.verbose:
                print("Metric '{}' did not improved for {} epochs. Early stop!".format(
                    self.metric, self.patience))
            self.trainer.early_stop()


class LambdaCallback(Callback):
    """Wrapper on callback that helps create callbacks quickly."""

    def __init__(self,
                 on_train_begin=None, on_train_end=None,
                 on_epoch_begin=None, on_epoch_end=None,
                 on_batch_begin=None, on_batch_end=None):

        self.on_train_begin = on_train_begin if on_train_begin is not None else lambda m: None
        self.on_train_end = on_train_end if on_train_end is not None else lambda m: None
        self.on_epoch_begin = on_epoch_begin if on_epoch_begin is not None else lambda e, m: None
        self.on_epoch_end = on_epoch_end if on_epoch_end is not None else lambda e, m: None
        self.on_batch_begin = on_batch_begin if on_batch_begin is not None else lambda b, m: None
        self.on_batch_end = on_batch_end if on_batch_end is not None else lambda b, m: None


class ModelCheckpoint(Callback):
    """Saves PyTorch module weights after epoch of training (if conditions are met)."""

    def __init__(self, path, metric='val_loss', save_best=False, verbose=0):
        """Initialize ModelCheckpoint callback.

        Args:
            path (str): Path where to save weights.
            metrics (str): Metric name to keep track of. (Default: 'val_loss')
            save_best (bool): If to save only best checkpoints according to given metric.
                (Default: False)
            verbose (int): Two levels of verbosity:
                * 1 - show update information,
                * 0 - show nothing. (<- Default)
        """

        self.path = path
        self.metric = metric
        self.save_best = save_best
        self.verbose = verbose

        self._best_value = float('inf')

    def on_epoch_end(self, epoch, metrics):
        if not self.save_best:
            self.trainer.save_ckpt(self.path)
            if self.verbose:
                print("Saving module state at: {}".format(self.path))
        else:
            if not self.metric in metrics:
                raise ValueError("There is no such metric evaluated: {}".format(self.metric))

            new_value = metrics[self.metric]
            if new_value < self._best_value:
                if self.verbose:
                    print("Saving new best module state at: {}".format(self.path))
                self._best_value = new_value
                self.trainer.save_ckpt(self.path)


class CallbackList(Callback):
    """Simplifies calling all callbacks from list."""

    def __init__(self, callbacks, trainer):
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.trainer = trainer

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, is_aborted):
        for callback in self.callbacks:
            callback.on_train_end(is_aborted)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch, metrics):
        for callback in self.callbacks:
            callback.on_batch_end(batch, metrics)


class TorchTrainer(object):
    """High-level toolbox to train, evaluate and infer PyTorch nn.Module."""

    def __init__(self, model):
        """Initialize TorchTrainer.

        Args:
            model (torch.nn.Module): PyTorch neural net module.
        """

        if not isinstance(model, nn.Module):
            raise ValueError("Model needs to inherit from torch.nn.Module!")

        self.model = model

        self._early_stop = False
        self._is_compiled = False

    def compile(self, optimizer, loss, metrics=None):
        """Set trainer optimizer, loss and metrics to evaluate.

        Args:
            optimizer (torch.optim.Optimizer): Parameters optimizer.
            loss (func): Loss function with signature `func(pred, target) -> scalar`, where
                `pred` is PyTorch module output and `target` is target labels/values.
            metrics (dict): List of functions with the same signature as loss. Those metrics will be
                evaluated on each training and validation batch. Final value is always average
                over all batches. (Default: always loss)
        """

        self.optim = optimizer
        self.loss = loss
        self.metrics = {'loss': loss}
        if metrics is not None:
            for metric in metrics:
                self.metrics[metric.__name__] = metric

        self._is_compiled = True

    def evaluate(self, data, target, batch_size=64, verbose=0):
        """Evaluate PyTorch module on dataset.

        Args:
            data (np.ndarray): Data to train on. First dimension must be batch dimension.
            target (np.ndarray): Targets to train on. First dimension must be batch dimension.
            batch_size (int): Single update data batch size. (Default: 64)
            verbose (int): Two levels of verbosity:
                * 1 - show progress bar,
                * 0 - show nothing. (<- Default)
        """

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(target)),
            batch_size=batch_size,
            shuffle=False
        )

        self.evaluate_loader(data_loader=data_loader, verbose=verbose)

    def evaluate_loader(self, data_loader, verbose=0):
        """Evaluate PyTorch module on dataset (using PyTorch DataLoader class).

        Args:
            data_loader (torch.utils.data.DataLoader): Train data loader.
            verbose (int): Two levels of verbosity:
                * 1 - show progress bar,
                * 0 - show nothing. (<- Default)

        Return:
            dict: Metrics evaluation results, averaged over all batches.
        """

        # Prepare for evaluation
        self._prepare()

        # Evaluate on whole dataset
        results_avg = defaultdict(float)
        for iter_t, (data, target) in enumerate(
                tqdm(data_loader, ascii=True, desc="Evaluate", disable=(not verbose))):
            results_tmp = self._eval_metrics(data, target)
            self._average_metrics(results_avg, results_tmp, iter_t)

        return results_avg

    def fit(self, data, target, batch_size=64, epochs=1, verbose=1, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, initial_epoch=0):
        """Fit PyTorch module to dataset.

        Args:
            data (np.ndarray): Data to train on. First dimension must be batch dimension.
            target (np.ndarray): Targets to train on. First dimension must be batch dimension.
            batch_size (int): Single update data batch size. (Default: 64)
            epochs (int): How many times iterate over whole dataset. (Default: 1)
            verbose (int): Two levels of verbosity:
                * 1 - show progress bar, (<- Default)
                * 0 - show nothing.
            callbacks (list of Callback): Event train, epoch and batch events listeners.
            validation_split (float): How big part of data put away for validation. (Default: 0.0)
            validation_data (tuple): Tuple: (val_data, val_targets). Overwrites `validation_split`.
            shuffle (bool): If randomize data order in each epoch. (Default: True)
            initial_epoch (int): From which number start to count epochs. (Default: 0)
        """

        # Get validation data
        if validation_data is not None:
            X_train, y_train = data, target
            X_val, y_val = validation_data
        elif validation_split > 0.0 and validation_split < 1.0:
            X_train, X_val, y_train, y_val = train_test_split(
                data, target, test_size=validation_split)
        else:
            X_train, y_train = data, target
            X_val, y_val = None, None

        # Create training data loader
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=shuffle
        )

        # Create validation data loader if there is given data
        validation_loader = None
        if X_val is not None and y_val is not None:
            validation_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                batch_size=batch_size,
                shuffle=False
            )

        self.fit_loader(data_loader=data_loader, validation_loader=validation_loader,
                        epochs=epochs, verbose=verbose, callbacks=callbacks,
                        initial_epoch=initial_epoch)

    def fit_loader(self, data_loader, epochs=1, verbose=1, callbacks=None,
                   validation_loader=None, initial_epoch=0):
        """Fit PyTorch module to dataset (using PyTorch DataLoader class).

        Args:
            data_loader (torch.utils.data.DataLoader): Train data loader.
            epochs (int): How many times iterate over whole dataset. (Default: 1)
            verbose (int): Two levels of verbosity:
                * 1 - show progress bar, (<- Default)
                * 0 - show nothing.
            callbacks (list of Callback): Event train, epoch and batch events listeners.
            validation_loader (torch.utils.data.DataLoader): Validation data loader. (Default: None)
            initial_epoch (int): From which number start to count epochs. (Default: 0)
        """

        # Prepare for training
        self._prepare()

        # Create callbacks list
        callbacks_list = CallbackList(callbacks, trainer=self)

        try:
            # Train for # epochs
            callbacks_list.on_train_begin()
            for epoch in range(initial_epoch, epochs):
                callbacks_list.on_epoch_begin(epoch)

                # Train on whole dataset
                results_avg = defaultdict(float)
                pbar = tqdm(data_loader, ascii=True,
                            desc="Epoch: {:2d}/{}".format(epoch + 1, epochs), disable=(not verbose))
                for iter_t, (data, target) in enumerate(pbar):
                    callbacks_list.on_batch_begin(iter_t)

                    self.optim.zero_grad()

                    results_tmp = self._eval_metrics(data, target)
                    self._average_metrics(results_avg, results_tmp, iter_t)

                    results_tmp['loss'].backward()
                    self.optim.step()

                    pbar.set_postfix(**results_avg)
                    callbacks_list.on_batch_end(iter_t, results_tmp)

                if validation_loader is not None:
                    results_val = self.evaluate_loader(validation_loader)
                    results_avg = self._merge_results(results_avg, results_val)
                    pbar.set_postfix(**results_avg)

                callbacks_list.on_epoch_end(epoch, results_avg)
                if self._early_stop:
                    break
            callbacks_list.on_train_end(False)
        except:
            callbacks_list.on_train_end(True)
            raise

    def save_ckpt(self, path):
        """Save model weights.

        Args:
            path(str): Path where to store weights.
        """

        torch.save(self.model.state_dict(), path)

    def load_ckpt(self, path):
        """Load model weights.

        Args:
            path(str): Path where to load weights from.
        """

        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        else:
            raise ValueError("Given path doesn't exist!")

    def early_stop(self):
        """Sets flag to stop current training after current epoch ends."""

        self._early_stop = True

    def _average_metrics(self, avg, tmp, iter_t):
        """Iteratively average metrics dictionary values.

        Args:
            avg (dict): Dictionary with average metrics values.
            tmp (dict): New metrics values to add to average.
            iter_t (int): Current iteration number (starting from 0).

        Note:
            `avg` dictionary values are changed in place.
        """

        for key, value in tmp.items():
            avg[key] += (value - avg[key]) / (iter_t + 1)

    def _eval_metrics(self, data, target):
        """Evaluate PyTorch module with all metrics.

        Args:
            data (np.ndarray): Batch of data examples.
            target (np.ndarray): Batch of true values/targets.

        Return:
            dict: Metrics evaluation results, keys are metrics' names.
        """

        results = {}
        for key, func in self.metrics.items():
            results[key] = func(self.model(data), target)

        return results

    def _merge_results(self, train_r, val_r):
        """Merge training and validation metrics evaluation results.

        It adds 'val_' prefix before validation results keys.

        Args:
            train_r (dict): Training metrics evaluation results.
            val_r (dict): Validation metrics evaluation results.

        Return:
            dict: Dictionary with merged metrics evaluation results.
        """

        merged_r = {**train_r}
        for key, value in val_r.items():
            merged_r['val_' + key] = value

        return merged_r

    def _prepare(self):
        """Prepares Trainer for training/evaluation."""

        self._early_stop = False
        if not self._is_compiled:
            raise ValueError("You need to compile Trainer before using it!")
