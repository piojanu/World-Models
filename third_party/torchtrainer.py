"""The MIT License
Copyright 2018 Piotr Januszewski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

import os.path

import torch
import torch.nn as nn
import torch.utils.data

from collections import defaultdict
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

    def on_batch_begin(self, batch, batch_size):
        pass

    def on_batch_end(self, batch, batch_size, metrics):
        pass


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

    def on_batch_begin(self, batch, batch_size):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, batch_size)

    def on_batch_end(self, batch, batch_size, metrics):
        for callback in self.callbacks:
            callback.on_batch_end(batch, batch_size, metrics)


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
            raise ValueError("\tThere is no such metric evaluated: {}".format(self.metric))

        new_value = metrics[self.metric]
        if new_value < self._best_value:
            if self.verbose:
                print("\tMetric '{}' improved from {} to {}".format(
                    self.metric, self._best_value, new_value))
            self._best_value = new_value
            self._last_epoch = epoch
        elif self.verbose:
            print("\tMetric '{}' did not improved. Current best: {}".format(
                self.metric, self._best_value))

        if self._last_epoch + self.patience <= epoch:
            if self.verbose:
                print("\tMetric '{}' did not improved for {} epochs. Early stop!".format(
                    self.metric, self.patience))
            self.trainer.early_stop()


class LambdaCallback(Callback):
    """Wrapper on callback that helps create callbacks quickly."""

    def __init__(self,
                 on_train_begin=None, on_train_end=None,
                 on_epoch_begin=None, on_epoch_end=None,
                 on_batch_begin=None, on_batch_end=None):

        self.on_train_begin = on_train_begin if on_train_begin is not None else lambda: None
        self.on_train_end = on_train_end if on_train_end is not None else lambda a: None
        self.on_epoch_begin = on_epoch_begin if on_epoch_begin is not None else lambda e: None
        self.on_epoch_end = on_epoch_end if on_epoch_end is not None else lambda e, m: None
        self.on_batch_begin = on_batch_begin if on_batch_begin is not None else lambda b, s: None
        self.on_batch_end = on_batch_end if on_batch_end is not None else lambda b, s, m: None


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
                print("\tSaving module state at: {}".format(self.path))
        else:
            if not self.metric in metrics:
                raise ValueError("\tThere is no such metric evaluated: {}".format(self.metric))

            new_value = metrics[self.metric]
            if new_value < self._best_value:
                if self.verbose:
                    print("\tSaving new best module state at: {}".format(self.path))
                self._best_value = new_value
                self.trainer.save_ckpt(self.path)


class MultiDataset(torch.utils.data.Dataset):
    """Multi input/output dataset."""

    def __init__(self, data, targets):
        """Initialize MultiDataset.

        Args:
            data (np.ndarray or list): List of arrays 'N x *' where 'N' is number of examples
                and '*' indicates any number of dimensions.
            target (np.ndarray or list): List of arrays 'N x *' where 'N' is number of examples
                and '*' indicates any number of dimensions.

        Note:
            Tensors should have the same size of the first dimension
        """

        if not isinstance(data, (list, tuple)):
            self.data = [torch.from_numpy(data)]
        else:
            self.data = [torch.from_numpy(d) for d in data]

        if not isinstance(targets, (list, tuple)):
            self.targets = [torch.from_numpy(targets)]
        else:
            self.targets = [torch.from_numpy(t) for t in targets]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.data), tuple(tensor[index] for tensor in self.targets)

    def __len__(self):
        return self.data[0].size(0)


class TorchTrainer(object):
    """High-level toolbox to train, evaluate and infer PyTorch nn.Module."""

    def __init__(self, model, device_name='cpu'):
        """Initialize TorchTrainer.

        Args:
            model (torch.nn.Module): PyTorch neural net module.
            device_name (str): The desired device of data/target tensors ('cpu' or 'cuda' for GPU).
                (Default: cpu)
        """

        if not isinstance(model, nn.Module):
            raise ValueError("Model needs to inherit from torch.nn.Module!")

        self.device = torch.device(device_name)
        self.model = model.to(self.device, non_blocking=True)

        self._early_stop = False
        self._is_compiled = False

    def compile(self, optimizer, loss, metrics=None):
        """Set trainer optimizer, loss and metrics to evaluate.

        Args:
            optimizer (torch.optim.Optimizer): Parameters optimizer.
            loss (func): Loss function with signature `func(pred, target) -> scalar tensor`, where
                `pred` is PyTorch module output (can be tuple if multiple outputs) and `target`
                is target labels/values from provided data (can be tuple if multiple targets).
            metrics (list): List of functions with the same signature as loss. Those metrics will be
                evaluated on each training and validation batch. Final value is always average
                over all batches. (Default: always loss)
        """

        self.optim = optimizer
        self.loss = loss
        self.metrics = {}
        if metrics is not None:
            for metric in metrics:
                self.metrics[metric.__name__] = metric

        self._is_compiled = True

    def evaluate(self, data, target, batch_size=64, verbose=0):
        """Evaluate PyTorch module on dataset.

        Args:
            data (np.ndarray or list): Data to evaluate on or list of multiple data arrays.
                Array shape should be 'N x *' where 'N' is number of examples and '*' indicates
                any number of dimensions. Its type should be the same as desired tensor.
            target (np.ndarray or list): Target to evaluate on or list of multiple target arrays.
                Array shape should be 'N x *' where 'N' is number of examples and '*' indicates
                any number of dimensions. Its type should be the same as desired tensor.
            batch_size (int): Single update data batch size. (Default: 64)
            verbose (int): Two levels of verbosity:
                * 1 - show progress bar,
                * 0 - show nothing. (<- Default)
        """

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            MultiDataset(data, target),
            batch_size=batch_size,
            shuffle=False
        )

        return self.evaluate_loader(data_loader=data_loader, verbose=verbose)

    def evaluate_loader(self, data_loader, verbose=0):
        """Evaluate PyTorch module on dataset (using PyTorch DataLoader class).

        Args:
            data_loader (torch.utils.data.DataLoader): Evaluation data loader.
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
            data = [d.to(self.device, non_blocking=True) for d in data]
            target = [t.to(self.device, non_blocking=True) for t in target]

            results_tmp, _ = self._eval_metrics(data, target)
            self._average_metrics(results_avg, results_tmp, iter_t)

        return results_avg

    def fit(self, data, target, batch_size=64, epochs=1, verbose=1, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, initial_epoch=0):
        """Fit PyTorch module to dataset.

        Args:
            data (np.ndarray or list): Data to train on or list of multiple data arrays.
                Array shape should be 'N x *' where 'N' is number of examples and '*' indicates
                any number of dimensions. Its type should be the same as desired tensor.
            target (np.ndarray or list): Target to train on or list of multiple target arrays.
                Array shape should be 'N x *' where 'N' is number of examples and '*' indicates
                any number of dimensions. Its type should be the same as desired tensor.
            batch_size (int): Single update data batch size. (Default: 64)
            epochs (int): How many times iterate over whole dataset. (Default: 1)
            verbose (int): Two levels of verbosity:
                * 1 - show progress bar, (<- Default)
                * 0 - show nothing.
            callbacks (list of Callback): Event train, epoch and batch events listeners.
            validation_split (float): How big fraction of data put away for validation evaluation.
                The validation set is selected from the last samples in the data and target.
                (Default: 0.0)
            validation_data (tuple): Tuple: (val_data, val_targets). Overwrites `validation_split`.
                (Default: None)
            shuffle (bool): If randomize data order in each epoch. (Default: True)
            initial_epoch (int): From which number start to count epochs. (Default: 0)
        """

        # Validation split
        if validation_data is None and validation_split > 0.0 and validation_split < 1.0:
            data, data_val = self._val_split(data, validation_split)
            target, target_val = self._val_split(target, validation_split)
            validation_data = (data_val, target_val)

        # Create training data loader
        data_loader = torch.utils.data.DataLoader(
            MultiDataset(data, target),
            batch_size=batch_size,
            shuffle=shuffle
        )

        # Create validation data loader if there is given data
        if validation_data is not None:
            validation_loader = torch.utils.data.DataLoader(
                MultiDataset(*validation_data),
                batch_size=batch_size,
                shuffle=False
            )
        else:
            validation_loader = None

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
                with tqdm(data_loader, ascii=True, desc="{:2d}/{}".format(epoch + 1, epochs),
                          disable=(not verbose)) as pbar:
                    for iter_t, (data, target) in enumerate(pbar):
                        callbacks_list.on_batch_begin(iter_t, data[0].size(0))
                        data = [d.to(self.device, non_blocking=True) for d in data]
                        target = [t.to(self.device, non_blocking=True) for t in target]

                        self.optim.zero_grad()

                        results_tmp, loss = self._eval_metrics(data, target)
                        self._average_metrics(results_avg, results_tmp, iter_t)

                        loss.backward()
                        self.optim.step()

                        callbacks_list.on_batch_end(iter_t, data[0].size(0), results_tmp)

                        if (iter_t + 1) == len(data_loader) and validation_loader is not None:
                            results_val = self.evaluate_loader(validation_loader)
                            results_avg = self._merge_results(results_avg, results_val)

                        pbar.set_postfix(results_avg)

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

    @staticmethod
    def _average_metrics(avg, tmp, iter_t):
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
            torch.Tensor: Module loss on given batch.
        """

        # Unpack if target is list with only one element
        if isinstance(target, (list, tuple)) and len(target) == 1:
            target = target[0]

        pred = self.model(*data)
        loss = self.loss(pred, target)

        results = {'loss': loss.item()}
        for key, func in self.metrics.items():
            results[key] = func(pred, target).item()

        return results, loss

    @staticmethod
    def _merge_results(train_r, val_r):
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

    @staticmethod
    def _val_split(data, split):
        """Splits provided data into two sets.

        Args:
            data (np.ndarray or list): Data np.ndarray (or list of data np.ndarray-s) to split.
            split (float): Value between 0.0 and 1.0. Fraction of examples split to second set.

        Return:
            np.ndarray or list: First dataset of size '(1 - split) * number of samples'.
            np.ndarray or list: Second dataset of size 'split * number of samples'.
        """

        if isinstance(data, (list, tuple)):
            return list(zip(*[TorchTrainer._val_split(d, split) for d in data]))
        else:
            split_point = int((1 - split) * len(data))
            return data[:split_point], data[split_point:]

    def _prepare(self):
        """Prepares Trainer for training/evaluation."""

        self._early_stop = False
        if not self._is_compiled:
            raise ValueError("You need to compile Trainer before using it!")


if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_digits

    NUM_CLASSES = 10
    USE_CUDA = torch.cuda.is_available()

    # Load data and divide into datasets
    data, target = load_digits(n_class=NUM_CLASSES, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, stratify=target)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train)

    # Build PyTorch module
    class Net(nn.Module):
        def __init__(self, input_size, num_classes, hidden_units=100):
            super(Net, self).__init__()

            self.h = nn.Linear(input_size, hidden_units)
            self.out = nn.Linear(hidden_units, num_classes)

            # Weights initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # 'n' is number of inputs to each neuron
                    n = len(m.weight.data[1])
                    # "Xavier" initialization
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                    m.bias.data.zero_()

        def forward(self, x):
            hidden = torch.relu(self.h(x))
            return self.out(hidden)

    # Instantiate Torch Trainer!
    net = TorchTrainer(Net(data.shape[1], NUM_CLASSES), device_name='cuda' if USE_CUDA else 'cpu')

    # Create accuracy metric
    def acc(pred, target):
        return torch.mean((torch.max(pred, 1)[1] == target).type(torch.FloatTensor))

    # Compile trainer
    net.compile(
        optimizer=optim.SGD(net.model.parameters(), lr=1e-3, momentum=0.9),
        loss=nn.CrossEntropyLoss(),
        metrics=[acc]
    )

    # Fit module to training data
    net.fit(
        # Cast to appropriate type as Tensors are created with `torch.from_numpy(...)`.
        # See documentation: https://pytorch.org/docs/stable/torch.html#torch.from_numpy
        X_train.astype(np.float32), y_train.astype(np.long),
        epochs=100,
        validation_data=(X_val.astype(np.float32), y_val.astype(np.long)),
        callbacks=[
            EarlyStopping(verbose=1),
            ModelCheckpoint("/tmp/best_digits.ckpt", save_best=True, verbose=1)
        ]
    )

    # Evaluate module on test data
    metrics = net.evaluate(X_test.astype(np.float32), y_test.astype(np.long), verbose=1)
    print("Final weights loss: {}, accuracy: {}".format(metrics['loss'], metrics['acc']))