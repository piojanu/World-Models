"""The MIT License
Copyright 2018 Piotr Januszewski, Mateusz Jabłoński

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

import os.path
import csv

import torch
import torch.nn as nn
import torch.utils.data

from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from tqdm import tqdm


class RandomBatchSampler(torch.utils.data.Sampler):
    """Samples batches order randomly, without replacement.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Batch size (number of examples)

    Note:
        Last batch is always composed of dataset tail.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        return iter([
            list(range(x * self.batch_size, (x + 1) * self.batch_size))
            for x in torch.randperm(self.__len__() - 1).tolist()
        ] + [list(range((self.__len__() - 1) * self.batch_size, len(self.data_source)))])

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


@contextmanager
def evaluate(module):
    """Switch PyTorch module to evaluation mode and then restore previous mode."""

    is_train = module.training
    try:
        module.eval()
        yield module
    finally:
        if is_train:
            module.train()


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

    def on_train_begin(self, initial_epoch):
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

    def on_train_begin(self, initial_epoch):
        for callback in self.callbacks:
            callback.on_train_begin(initial_epoch)

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
    """Stops training if given metrics doesn't improve.

    Args:
        metric (str): Metric name to keep track of. (Default: 'val_loss')
        patience (int): After how many epochs without improvement training should stop.
            (Default: 5)
        verbose (int): Two levels of verbosity:
            * 1 - show update information,
            * 0 - show nothing. (<- Default)
    """

    def __init__(self, metric='val_loss', patience=5, verbose=0):
        self.metric = metric
        self.patience = patience
        self.verbose = verbose

        self._best_value = float('inf')
        self._last_epoch = None

    def on_train_begin(self, initial_epoch):
        self._last_epoch = initial_epoch

    def on_epoch_end(self, epoch, metrics):
        if self.metric not in metrics:
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

        self.on_train_begin = on_train_begin if on_train_begin is not None else lambda i: None
        self.on_train_end = on_train_end if on_train_end is not None else lambda a: None
        self.on_epoch_begin = on_epoch_begin if on_epoch_begin is not None else lambda e: None
        self.on_epoch_end = on_epoch_end if on_epoch_end is not None else lambda e, m: None
        self.on_batch_begin = on_batch_begin if on_batch_begin is not None else lambda b, s: None
        self.on_batch_end = on_batch_end if on_batch_end is not None else lambda b, s, m: None


class ModelCheckpoint(Callback):
    """Saves PyTorch module weights after epoch of training (if conditions are met).

    Args:
        path (str): Path where to save weights.
        metrics (str): Metric name to keep track of. (Default: 'val_loss')
        save_best (bool): If to save only best checkpoints according to given metric.
            (Default: False)
        verbose (int): Two levels of verbosity:
            * 1 - show update information,
            * 0 - show nothing. (<- Default)
    """

    def __init__(self, path, metric='val_loss', save_best=False, verbose=0):
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
            if self.metric not in metrics:
                raise ValueError("\tThere is no such metric evaluated: {}".format(self.metric))

            new_value = metrics[self.metric]
            if new_value < self._best_value:
                if self.verbose:
                    print("\tSaving new best module state at: {}".format(self.path))
                self._best_value = new_value
                self.trainer.save_ckpt(self.path)


class CSVLogger(Callback):
    """Saves metrics values to csv file after epoch of training.

    Args:
        filename (string): Filename where to log.
    """

    def __init__(self, filename):
        self.filename = filename

    def on_epoch_end(self, epoch, metrics):
        append_headers = False
        metrics = {'epoch': epoch, **metrics}
        if not os.path.exists(self.filename):
            append_headers = True
        with open(self.filename, "a") as f:
            w = csv.DictWriter(f, metrics.keys())
            if append_headers:
                w.writeheader()
            w.writerow(metrics)


class TensorBoardLogger(Callback):
    """Logging in TensorBoard without TensorFlow ops.

    Args:
        log_dir (string): Directory where to log.
    """

    Summary = None
    FileWriter = None

    def __init__(self, log_dir):
        if self.Summary is None or self.FileWriter is None:
            import tensorflow as tf
            self.Summary = tf.Summary
            self.FileWriter = tf.summary.FileWriter

        self.writer = self.FileWriter(log_dir)

    def on_epoch_end(self, epoch, metrics):
        for tag, value in metrics.items():
            self.writer.add_summary(
                self.Summary(value=[self.Summary.Value(tag=tag, simple_value=value)]),
                epoch)
        self.writer.flush()


class MultiDataset(torch.utils.data.Dataset):
    """Multi input/output dataset.

    Args:
        data (np.ndarray or list): List of arrays 'N x *' where 'N' is number of examples
            and '*' indicates any number of dimensions.
        targets (np.ndarray or list): List of arrays 'N x *' where 'N' is number of examples
            and '*' indicates any number of dimensions.

    Note:
        Tensors should have the same size of the first dimension
    """

    def __init__(self, data, targets):
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
    """High-level toolbox to train, evaluate and infer PyTorch nn.Module.


    Args:
        model (torch.nn.Module): PyTorch neural net module.
        device_name (str): The desired device of data/target tensors ('cpu' or 'cuda' for GPU).
            (Default: cpu)
    """

    def __init__(self, model, device_name='cpu'):
        if not isinstance(model, nn.Module):
            raise ValueError("Model needs to inherit from torch.nn.Module!")

        self.device = torch.device(device_name)
        self.model = model.to(self.device, non_blocking=True)

        self._early_stop = False
        self._is_compiled = False

    def compile(self, optimizer, loss, metrics=None, loss_weights=None):
        """Set trainer optimizer, loss and metrics to evaluate.

        Args:
            optimizer (torch.optim.Optimizer): Parameters optimizer.
            loss (func or dict): Loss function with signature `func(pred, target) -> scalar tensor`,
                where `pred` is PyTorch module output (can be tuple if multiple outputs) and
                `target` is target labels/values from provided data (can be tuple if multiple
                targets). Also, can be dictionary of such functions. Then outputs have to be
                OrderedDict. Outputs are then mapped to losses based on keys (strings) and to
                targets based on order.
            metrics (list or dict): List of functions with the same signature as loss. Those metrics
                will be evaluated on each training and validation batch. Final value (after epoch)
                is always average over all batches. Can be dictionary of such lists. Look above in
                loss docstring for mapping specifics. (Default: None)
            loss (dict): Dictionary specifying scalar coefficients (floats) to weight the loss
                contributions of different model outputs. Outputs are then mapped based on keys
                (strings). (Default: None)
        """

        self.optim = optimizer
        self.loss = loss
        self.metrics = metrics or ({} if isinstance(loss, dict) else [])
        self.loss_weights = loss_weights or \
            ({k: 1. for k in loss.keys()} if isinstance(loss, dict) else None)

        assert isinstance(self.loss, dict) == isinstance(self.metrics, dict), \
            "Both loss and metrics have to be dict or not together."

        self._is_multi_loss = isinstance(loss, dict)
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
        with tqdm(data_loader, ascii=True, disable=(not verbose), bar_format='{n_fmt}/{total_fmt} [{bar}] ETA: {remaining}, {rate_fmt}{postfix}') as pbar:
            for iter_t, (data, target) in enumerate(pbar):
                data = [d.to(self.device, non_blocking=True) for d in data]
                target = [t.to(self.device, non_blocking=True) for t in target]

                with torch.no_grad(), evaluate(self.model) as model:
                    pred = model(*data)
                _, results_tmp = self._eval_loss_n_metrics(pred, target)
                self._average_metrics(results_avg, results_tmp, iter_t)
                pbar.set_postfix({k: "{:.4f}".format(v) for k, v in results_avg.items()})

        return dict(results_avg)

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
            epochs (int): Final epoch after which training stops. (Default: 1)
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
            epochs (int): Final epoch after which training stops. (Default: 1)
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
        callbacks_list = CallbackList(callbacks or [], trainer=self)

        try:
            # Train for # epochs
            callbacks_list.on_train_begin(initial_epoch)
            for epoch in range(initial_epoch, epochs):
                callbacks_list.on_epoch_begin(epoch)

                # Train on whole dataset
                results_avg = defaultdict(float)
                if verbose:
                    tqdm.write('Epoch {:2d}/{}'.format(epoch + 1, epochs))
                with tqdm(data_loader, ascii=True, disable=(not verbose), bar_format='{n_fmt}/{total_fmt} [{bar}] ETA: {remaining}, {rate_fmt}{postfix}') as pbar:
                    for iter_t, (data, target) in enumerate(pbar):
                        callbacks_list.on_batch_begin(iter_t, data[0].size(0))
                        data = [d.to(self.device, non_blocking=True) for d in data]
                        target = [t.to(self.device, non_blocking=True) for t in target]

                        self.optim.zero_grad()

                        pred = self.model(*data)
                        loss, results_tmp = self._eval_loss_n_metrics(pred, target)
                        self._average_metrics(results_avg, results_tmp, iter_t)

                        loss.backward()
                        self.optim.step()

                        callbacks_list.on_batch_end(iter_t, data[0].size(0), results_tmp)

                        if (iter_t + 1) == len(data_loader) and validation_loader is not None:
                            results_val = self.evaluate_loader(validation_loader)
                            results_avg = self._merge_results(results_avg, results_val)

                        pbar.set_postfix({k: "{:.4f}".format(v) for k, v in results_avg.items()})

                callbacks_list.on_epoch_end(epoch, results_avg)
                if self._early_stop:
                    break
            callbacks_list.on_train_end(False)
        except Exception as e:
            callbacks_list.on_train_end(True)
            raise e

    def save_ckpt(self, path):
        """Save model weights.

        Args:
            path(string): Path where to store weights.
        """

        torch.save(self.model.state_dict(), path)

    def load_ckpt(self, path):
        """Load model weights.

        Args:
            path(string): Path where to load weights from.
        """

        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
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

    def _eval_loss_n_metrics(self, pred, target):
        """Evaluate PyTorch module with all metrics.

        Args:
            pred (object): NN module output, see `compile` for possible types.
            target (np.ndarray): Batch of true values/targets.

        Return:
            torch.Tensor: Module loss on given batch. In multi loss case it's sum of partial losses.
            dict: Metrics evaluation results, keys are metrics' names.
        """

        results = {}

        if self._is_multi_loss:
            assert isinstance(pred, OrderedDict), \
                "For multi loss case module output have to be OrderedDict."

            losses = []
            for (key, output), label in zip(pred.items(), target):
                losses.append(self.loss[key](output, label) * self.loss_weights[key])
                results[key + "_loss"] = losses[-1].item()

                if key not in self.metrics:
                    continue

                for func in self.metrics[key]:
                    results[key + "_" + func.__name__] = func(output, label).item()

            loss = sum(losses)
        else:
            # Unpack if target is list with only one element
            if len(target) == 1:
                target = target[0]

            loss = self.loss(pred, target)
            for func in self.metrics:
                results[func.__name__] = func(pred, target).item()

        results['loss'] = loss.item()
        return loss, results

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


def test_one_loss():
    import numpy as np
    import torch.optim as optim
    import pytest

    np.random.seed(7)
    torch.manual_seed(7)

    X = np.random.rand(100, 2).astype(np.float32)
    y = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.h = nn.Linear(2, 2)
            self.out = nn.Linear(2, 1)

        def forward(self, x):
            hidden = torch.tanh(self.h(x))
            return self.out(hidden)

    def l2(pred, target):
        return sum((target - pred)**2)

    net = TorchTrainer(Net())
    net.compile(
        optimizer=optim.SGD(net.model.parameters(), lr=1e-3, momentum=0.9),
        loss=nn.MSELoss(),
        metrics=[l2]
    )
    net.fit(X, y[:, np.newaxis], epochs=25, verbose=0)

    metrics = net.evaluate(X, y[:, np.newaxis], verbose=0)
    assert np.allclose(metrics["loss"], 0.135, atol=1.e-3)
    assert np.allclose(metrics["l2"], 6.708, atol=1.e-3)


def test_multi_loss():
    import numpy as np
    import torch.optim as optim
    import pytest

    from collections import OrderedDict

    np.random.seed(7)
    torch.manual_seed(7)

    X = np.random.rand(100, 2).astype(np.float32)
    y_1 = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]
    y_2 = X[:, 0] + X[:, 1]

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.h = nn.Linear(2, 3)
            self.out_1 = nn.Linear(3, 1)
            self.out_2 = nn.Linear(3, 1)

        def forward(self, x):
            hidden = torch.tanh(self.h(x))
            return OrderedDict([("out_1", self.out_1(hidden)), ("out_2", self.out_2(hidden))])

    def l2(pred, target):
        return sum((target - pred)**2)

    net = TorchTrainer(Net())
    net.compile(
        optimizer=optim.SGD(net.model.parameters(), lr=1e-3, momentum=0.9),
        loss={"out_1": nn.MSELoss(), "out_2": nn.L1Loss()},
        metrics={"out_1": [l2], "out_2": [l2]}
    )
    net.fit(X, [y_1[:, np.newaxis], y_2[:, np.newaxis]], epochs=40, verbose=0)

    metrics = net.evaluate(X, [y_1[:, np.newaxis], y_2[:, np.newaxis]], verbose=0)
    assert np.allclose(metrics["out_1_loss"], 0.116, atol=1.e-3)
    assert np.allclose(metrics["out_2_loss"], 0.283, atol=1.e-3)
    assert np.allclose(metrics["out_1_l2"], 5.771, atol=1.e-3)
    assert np.allclose(metrics["out_2_l2"], 5.494, atol=1.e-3)
    assert np.allclose(metrics["loss"], 0.399, atol=1.e-3)


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
            ModelCheckpoint("/tmp/best_digits.ckpt", save_best=True, verbose=1),
            TensorBoardLogger("/tmp/tensorboard")
        ]
    )

    # Evaluate module on test data
    eval_metrics = net.evaluate(X_test.astype(np.float32), y_test.astype(np.long), verbose=1)
    print("Final weights loss: {}, accuracy: {}".format(eval_metrics['loss'], eval_metrics['acc']))
