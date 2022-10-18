#######################
### Library imports ###
#######################

# standard library
import copy
import os
import sys

# data packages
import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline

# parallelization
from joblib import Parallel, delayed


########################
### Helper functions ###
########################

# index rows for both dataframes and arrays
def _subset_rows(x, idx):
    if isinstance(x, pd.DataFrame):
        return x.iloc[idx]
    else:
        return x[idx]


# index columns for both dataframes and arrays
def _subset_cols(x, idx):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, idx]
    else:
        return x[:, idx]


# alternative to sklearn's _fit_binary
# omits _ConstantPredictor and supports fit_params
def _fit_binary(estimator, X, y=None, **fit_params):
    estimator = sklearn.base.clone(estimator)
    estimator.fit(X=X, y=y, **fit_params)
    return estimator


class SmoothCrossEntropyLoss(nn.modules.loss._WeightedLoss):
    """
    Computes smoothed cross entropy (log) loss.
    Label smoothing works by clipping the true label values based on a
    specified smoothing parameter, e.g., with smoothing == 0.001 and n_classes == 2,
    [0, 1] --> [0.005, 0.995].
    The formula is given by label smoothed y = y * (1 - smoothing) + smoothing / n_classes
    This method can help prevent models from becoming over-confident.
    See paper: https://papers.nips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf
    """
    def __init__(self, weight=None, reduction="mean", smoothing=0.001, device="cpu"):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.device = device

    @staticmethod
    def _smooth(targets, n_classes, smoothing, device):
        """Helper for computing smoothed label values."""
        assert 0 <= smoothing <= 1
        with torch.no_grad():
            targets = (
                targets * (1 - smoothing)
                + torch.ones_like(targets).to(device) * smoothing / n_classes
            )
        return targets

    def forward(self, inputs, targets, sample_weight=None):
        # smooth targets
        targets = SmoothCrossEntropyLoss()._smooth(
            targets, 2, self.smoothing, self.device
        )
        # weight class predictions
        if self.weight is not None:
            inputs = inputs * self.weight.unsqueeze(0)

        if sample_weight is None:
            # binary_cross_entropy_with_logits returns mean log loss
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        else:
            # binary_cross_entropy_with_logits returns
            # [# obs., # classes] tensor of log losses
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            assert loss.size(0) == sample_weight.size(0)
            # compute weighted mean for each target
            loss = torch.sum(loss * sample_weight, dim=0) / torch.sum(sample_weight)
            # compute column-wise mean
            loss = torch.mean(loss)

        return loss


##############
### Models ###
##############

# Sub-class nn.Sequential to add reset_parameters method
class Sequential(nn.Sequential):
    def reset_parameters(self):
        for layer in self.children():
           if hasattr(layer, "reset_parameters"):
               layer.reset_parameters()



class Network(sklearn.base.BaseEstimator):
    """An sklearn-compatible wrapper for pytorch estimators.
    Wraps pytorch training and prediction in sklearn-compatible estimator with `fit` and
    `predict` methods and limited support for commonly-tuned net parameters. Supports
    early stopping and `eval_set` similarly to the LightGBM sklearn implementation.
    Parameters
    ----------
    net_obj : obj
        The instantiated pytorch network object to be used in training. Should have type
        that is a subclass of nn.Module.
    seed : int, optional
        Seed to be used for randomness in network initalization for reproducibility.
    optimizer : type, optional, default=torch.optim.Adam
        The optimizer class to be used in training. Should be a subclass of
        torch.optim.Optimizer.
    loss_fn : callable, optional, default=nn.BCEWithLogitsLoss()
        A function or callable loss object with signature `f(y_pred, y_true)`. If
        training with sample weights, should also accept a `sample_weight` argument.
    device : {"cpu", "cuda"}, optional, default="cpu"
        The device used in training.
    lr : float, optional, default=0.001
        The learning rate. Ignored if `lr_scheduler` is provided.
    weight_decay : float, optional, default=0
        Weight decay parameter used for network weight regularization.
    batch_size : int, optional, default=128
        Batch sized used in training.
    max_epochs : int, optional, default=10
        Maximum number of epochs used in training. Actual number of epochs used may be
        lower if early stopping is enabled.
    lr_scheduler : type, optional
        Learning rate scheduler for training, e.g. a class from
        torch.optim.lr_scheduler.
    lr_scheduler_params : dict, optional
        The parameters used to initialize the `lr_scheduler`.
    Attributes
    ----------
    self.metric_history_ : list of dict
        A list of dictionaries recording the values for each metric, eval_set, and
        epoch.
    self.early_stopping_history_ : list of float
        The list of values for only the first metric and eval_set, used for early
        stopping if specified.
    self.early_stopping_epoch_ : int or None
        The optimal epoch chosen by early stopping.
    self.net_ : obj
        The trained `net_obj`, which is used for prediction.
    self.metric_history_df_ : pd.DataFrame
        A dataframe wrapper around `self.metric_history_`.
    """

    def __init__(
        self,
        net_obj,
        seed=None,
        optimizer=torch.optim.Adam,
        loss_fn=None,
        device="cpu",
        lr=0.001,
        weight_decay=0,
        batch_size=128,
        max_epochs=10,
        lr_scheduler=None,
        lr_scheduler_params=None,
    ):

        self.net_obj = net_obj
        self.seed = seed
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_metric=None,
        patience=None,
        min_delta=None,
        verbose=False,
    ):
        """Trains the network, with support for early stopping.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training features.
        y : array-like, shape (n_samples, n_labels)
            The training labels.
        sample_weight : array-like, shape (n_samples, ), optional
            An optional array of sample weights to be used in computing loss, if
            desired, in which case `self.loss_fn` should accept a `sample_weight`
            argument.
        eval_set : list of tuple, optional
            A list of (X, y) tuples to be used for computing eval loss. At least one
            dataset is required for early stopping, in which case the first tuple in the
            list is used for early stopping evaluation.
        eval_names : list, optional
            An optionl list of the same length as `eval_set`, specifying the name of
            each dataset.
        eval_sample_weight : list of array-like, optional
            An optional list of the same length as `eval_set`, containing the sample
            weight array for each eval set (if `sample_weight` is in use).
        eval_metric : list of callable, optional
            A list of metric functions to be used in evaluation. Loss will be recorded
            for all metrics, but only the first metric provided will be used for early
            stopping, if enabled. Should have signature `f(y_pred, y_true), with a
            `sample_weight` parameter if weights are being used.
        patience : int, optional
            The number of epochs of increasing loss tested before stopping early. The
            number to be tested is reset after every epoch with a decrease in loss. This
            parameter may be overridden if `min_delta` is also set.
        min_delta : float, optional
            The minimum decrease in loss required to continue training. If loss doss not
            decrease by more than this value, training will be stopped early and the
            stopping epoch will be recorded in the `early_stopping_epoch_` attribute.
        verbose : int or bool, optional, default=False
            If False, no loss is printed during training. Otherwise, results are printed
            after every `verbose` epochs.
        Returns
        -------
        self : obj
            Returns the estimator itself, in keeping with sklearn requirements.
        """

        # initialize device for training
        device = torch.device(self.device)

        # set seed for network weight initialization
        if self.seed is not None:
            torch.manual_seed(self.seed)
            # this should be redundant with torch.manual_seed
            torch.cuda.manual_seed_all(self.seed)
        # send pytorch network to specified device
        net = self.net_obj.to(device)
        # reset initial parameters for net
        net.reset_parameters()
        # initialize loss and optimizer
        if self.loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = self.loss_fn
        optimizer = self.optimizer(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # helper for converting to tensor
        def to_tensor(a):
            return torch.tensor(a, dtype=torch.float32).to(device)

        if sample_weight is not None:
            # add sample_weight as column in X to pass through the DataLoader for
            # creating batches
            X_w_weights = np.append(X, sample_weight, 1)
            X_nn = to_tensor(X_w_weights)
        else:
            X_nn = to_tensor(X)

        y_nn = to_tensor(y)

        # Add extra dimension if y is single dimensional
        if len(y_nn.shape) == 1:
            y_nn = y_nn.unsqueeze(1)

        # set up for evaluation on train/val data
        if eval_set is None:
            eval_set = []
        if eval_metric is None:
            eval_metric = []
        if eval_names is None:
            eval_names = [f"eval_{i}" for i in range(len(eval_set))]
        if eval_sample_weight is None:
            eval_sample_weight = [None] * len(eval_set)

        # add train data as an eval set
        eval_set.append((X, y))
        eval_sample_weight.append(sample_weight)
        eval_names.append("train")

        # convert eval sets to tensors
        eval_set = [(to_tensor(tup[0]), to_tensor(tup[1])) for tup in eval_set]
        # convert eval weights to tensors
        eval_sample_weight = [
            to_tensor(weight) if weight is not None else weight
            for weight in eval_sample_weight
        ]

        eval_metric = [
            m if isinstance(m, tuple) else (f"metric_{i}", m)
            for i, m in enumerate(eval_metric)
        ]
        eval_metric.append(("objective", loss_fn))

        # set up dataloader for batches
        dataset = torch.utils.data.TensorDataset(X_nn, y_nn)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        lr_scheduler = self.lr_scheduler
        if self.lr_scheduler_params is None:
            lr_scheduler_params = {}
        else:
            lr_scheduler_params = self.lr_scheduler_params

        if lr_scheduler == "OneCycleLR":
            one_cycle_lr = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                epochs=self.max_epochs,
                steps_per_epoch=len(dataloader),
                **lr_scheduler_params,
            )
        if lr_scheduler == "ReduceLROnPlateau":
            reduce_lr_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **lr_scheduler_params
            )

        # track number of epochs with increasing loss for early stopping
        self.metric_history_ = []
        self.early_stopping_history_ = []
        self.early_stopping_epoch_ = None
        min_valmetric = np.inf
        increases = 0
        best_params = copy.deepcopy(net.state_dict())
        for epoch in range(self.max_epochs):

            # construct batches
            for batch_x, batch_y in dataloader:
                if sample_weight is not None:
                    # extract last column of batch_x (the sample weights)
                    batch_sample_weight = batch_x[:, -1].reshape(-1, 1)
                    batch_x = batch_x[:, :-1]
                else:
                    batch_sample_weight = None

                net.train()

                # zero gradients to start
                optimizer.zero_grad()

                output = net.forward(batch_x)
                if sample_weight is not None:
                    loss = loss_fn(output, batch_y, sample_weight=batch_sample_weight)
                else:
                    loss = loss_fn(output, batch_y)

                loss.backward()
                optimizer.step()
                if lr_scheduler == "OneCycleLR":
                    one_cycle_lr.step()

            net.eval()

            # record metrics for each eval set
            for i in range(len(eval_set)):
                X_val, y_val = eval_set[i]
                # Add extra dimension if y is single dimensional
                if len(y_val.shape) == 1:
                    y_val = y_val.unsqueeze(1)

                eval_sample_weight_ = eval_sample_weight[i]
                set_name = eval_names[i]
                with torch.no_grad():
                    preds = net(X_val)
                for j in range(len(eval_metric)):
                    metric_name, metric_fn = eval_metric[j]
                    if sample_weight is not None:
                        metric_val = metric_fn(
                            preds, y_val, sample_weight=eval_sample_weight_
                        )
                    else:
                        metric_val = metric_fn(preds, y_val)
                    if isinstance(metric_val, torch.Tensor):
                        metric_val = metric_val.item()
                    row = {
                        "epoch": epoch,
                        "data": set_name,
                        "metric": metric_name,
                        "value": metric_val,
                    }
                    self.metric_history_.append(row)
                    # use first val set and first metric for early stopping
                    if i == 0 and j == 0:
                        self.early_stopping_history_.append(metric_val)

            if verbose:
                verbose = int(verbose)
                if epoch % verbose == 0:
                    for d in self.metric_history_[-len(eval_set) * len(eval_metric) :]:
                        print(d)

            # if val set is present, record history and follow early stopping parameters
            if len(eval_set) > 1:
                val_metric = self.early_stopping_history_[-1]
                if lr_scheduler == "ReduceLROnPlateau":
                    reduce_lr_on_plateau.step(val_metric)

                # early stopping based on minimum decrease in loss
                if min_delta is not None and epoch > 0:
                    if self.early_stopping_history_[-2] - val_metric < min_delta:
                        print("Early stopping at epoch ", epoch)
                        self.early_stopping_epoch_ = epoch
                        break

                # early stopping based on number of epochs with increasing loss
                if val_metric < min_valmetric:
                    min_valmetric = val_metric
                    increases = 0
                    # save model paramaters for current best epoch
                    best_params = copy.deepcopy(net.state_dict())
                elif patience is not None:
                    increases += 1
                    if increases > patience:
                        print("Early stopping at epoch ", epoch)
                        self.early_stopping_epoch_ = epoch
                        break

        # if using early stopping, reload net with best params
        if patience is not None or min_delta is not None:
            # load model paramaters from best epoch
            net.load_state_dict(best_params)
            net.eval()

        # store network for prediction
        self.net_ = net

        self.metric_history_df_ = pd.DataFrame(self.metric_history_)

        return self

    def predict(self, X):
        """Predicts using the trained network.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            An array of the same shape as the one used in training, containing the data
            to be used for predictions.
        Returns
        -------
        np.array
            Model predictions in an array of shape (n_samples, n_labels).
        """

        # cast to tensor and move to device
        device = torch.device(self.device)
        X_nn = torch.tensor(X, dtype=torch.float32).to(device)

        # forward pass through network for predictions
        # return predictions as numpy array
        with torch.no_grad():
            return self.net_(X_nn).cpu().detach().numpy().astype("float32")

    def predict_proba(self, X):
        """Return predictions on probability scale for classification network.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            An array of the same shape as the one used in training, containing the data
            to be used for predictions.
        Returns
        -------
        np.array
            Model predictions in an array of shape (n_samples, n_labels), with sigmoid
            transformation applied (i.e. predicted probabilities).
        """

        preds = self.predict(X)
        preds_proba = 1 / (1 + np.exp(-preds))
        return preds_proba.astype("float32")


class CVModel(sklearn.base.BaseEstimator):
    """Wraps an sklearn estimator to train in folds using cross validation.
    In addition to training separate models for each fold, this estimator allows for
    predictions on new data or out-of-fold predictions on the training data. This makes
    it a convenient way to get out-of-sample predictions for every observation in a
    train set
    Parameters
    ----------
    model : obj
        Sklearn-compatible estimator to be used in training. To work with the CV
        procedure, the estimator must accept an `eval_set` parameter, although the
        implementation may be to just accept this parameter and do nothing with it.
    Attributes
    ----------
    splits_ : list of tuple
        The CV splits used in training, in the form of a list of `(idx_train, idx_test)`
        tuples.
    models_ : list of obj
        A list of trained models of same length as `self.splits_`, with each model cloned
        from `self.model` and trained on the corresponding split in `self.splits_`.
    """

    def __init__(self, model):
        self.model = model

    def fit(
        self,
        X,
        y=None,
        splits=None,
        sample_weight=None,
        eval_weights=False,
        **fit_params,
    ):
        """Split data into folds and train separate model for each split.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The features for the training data, which will be split into folds for
            separate model fitting.
        y : array-like, shape (n_samples, n_labels), optional
            The training labels, if required.
        splits : list of tuple
            The CV splits to  be used in training, in the form of a list of `(idx_train,
            idx_test)` tuples.
        sample_weight : array-like, shape (n_samples, )
            Sample weights for each observation in the training data. If provided, these
            will be split along with training features/labels and passed to each fold's
            model as a fit parameter.
        eval_weights : bool
            If True, pass the `sample_weight` to each model for eval sets also, as an
            `eval_sample_weight` fit parameter.
        Returns
        -------
        self : obj
            Returns the estimator itself, in keeping with sklearn requirements.
        """

        if splits is None:
            splits = [None]

        self.splits_ = splits

        # ensure indexing is same as rows if inputs are dataframes
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
        if isinstance(y, pd.DataFrame):
            y = y.reset_index(drop=True)

        # for each fold, split data, train model, and store it
        self.models_ = []
        for i, (idx_train, idx_val) in enumerate(splits):
            if 'verbose' in fit_params and fit_params['verbose']:
                print(f"Fitting split {i+1}/{len(splits)}")

            m = sklearn.base.clone(self.model)

            # split sample weights if they are included
            if sample_weight is not None:
                fit_params["sample_weight"] = _subset_rows(sample_weight, idx_train)
                if eval_weights:
                    fit_params["eval_sample_weight"] = _subset_rows(sample_weight, idx_val)

            m.fit(
                _subset_rows(X, idx_train),
                _subset_rows(y, idx_train),
                eval_set=[(_subset_rows(X, idx_val), _subset_rows(y, idx_val))],
                **fit_params,
            )
            self.models_.append(m)

        return self

    def _predict(self, X, use_splits=False, predict_proba=True, **pred_params):
        """Private method for implementing `use_splits` and `predict_proba`."""

        # use training splits to prediction out-of-fold
        if use_splits:
            order = []
            preds_shuffled = []
            for m, split_indices in zip(self.models_, self.splits_):
                oof_indices = split_indices[1]
                # get raw or transformed preds, as specified
                if predict_proba:
                    preds_oof = m.predict_proba(_subset_rows(X, oof_indices), **pred_params)
                else:
                    preds_oof = m.predict(_subset_rows(X, oof_indices), **pred_params)
                # record the prediction and original index for each observation
                order.append(oof_indices)
                preds_shuffled.append(preds_oof)
            # fix the order to match the input data
            order = np.concatenate(order)
            preds_shuffled = np.concatenate(preds_shuffled)
            preds = np.empty_like(preds_shuffled)
            preds[order] = preds_shuffled
        # predict using average of all k models
        else:
            preds = []
            for m in self.models_:
                if predict_proba:
                    pred = m.predict_proba(X, **pred_params)
                else:
                    pred = m.predict(X, **pred_params)
                preds.append(pred)
            preds = np.mean(preds, axis=0)

        return preds

    def predict(self, X, use_splits=False, **pred_params):
        """Predict on input data using trained models.
        By setting `use_splits` to `True`, one can immediately obtain an out-of-sample
        prediction for each training observation by using only the model for which the
        observation was out-of-fold. Otherwise, predictions are averaged across the
        models from each fold.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            An array of the same shape as the one used in training, containing the data
            to be used for predictions.
        use_splits : bool
            Whether or not to use training splits to get out-of-fold predictions. Only
            applicable if `X` is the original data used in training.
        **pred_params
            Additional prediction parameters to be passed to the models.
        Returns
        -------
        array-like
            Model predictions in an array of shape (n_samples, n_labels).
        """

        return self._predict(
            X, use_splits=use_splits, predict_proba=False, **pred_params
        )

    def predict_proba(self, X, use_splits=False, **pred_params):
        """Get predictions on probability-scale using trained models.
        By setting `use_splits` to `True`, one can immediately obtain an out-of-sample
        prediction for each training observation by using only the model for which the
        observation was out-of-fold. Otherwise, predictions are averaged across the
        models from each fold.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            An array of the same shape as the one used in training, containing the data
            to be used for predictions.
        use_splits : bool
            Whether or not to use training splits to get out-of-fold predictions. Only
            applicable if `X` is the original data used in training.
        **pred_params
            Additional prediction parameters to be passed to the models.
        Returns
        -------
        array-like
            Model predictions in an array of shape (n_samples, n_labels), with sigmoid
            transformation applied (i.e. predicted probabilities).
        """

        return self._predict(
            X, use_splits=use_splits, predict_proba=True, **pred_params
        )


class SeedCVModel(sklearn.base.BaseEstimator):
    """
    Runs `CVModels` multiple times with different seeds and averages
    the prediction. Each seed is used to simultaneously create CV splits
    and set any non-deterministic values for the model (meaning that
    model configurations may differ across the CV runs).
    """
    def __init__(self, model, seeds):
        self.model = model
        self.seeds = seeds

    def fit(self, X, y, split_method, n_folds, groups, **fit_params):
        """
        Loops over seeds and creates CV splits and fits CVModel` for each
        seed.
        Split method is one of one of {"grouped", "target stratified"}.
        """
        models = []
        for seed in self.seeds:
            # copy model with new seed
            model = sklearn.base.clone(self.model).set_params(seed=seed)
            cv = CVModel(model)
            # make splits based on seed
            splits = kfold_splits(
                split_method, # one of {"grouped", "target stratified"}
                n_folds=n_folds,
                groups=groups,
                random_state=seed,
                X=X,
                y=y,
            )
            cv.fit(
                X=X,
                y=y,
                splits=splits,
                **fit_params,
            )
            models.append(cv)

        self.models = models

        return self

    def predict_proba(self, X, use_splits=False, **pred_params):
        """
        Predicts the probability by averaging probability
        predictions with models trained with different seeds.
        When `use_splits==True`, for a given `CVModel`, preds for each obs.
        are made with the model trained with that obs. in the out-of-fold sample.
        When `use_splits==False`, for a given `CVModel`, preds are averaged over
        the model for each fold.
        """
        preds = []
        # get preds for each model
        for m in self.models:
            pred = m.predict(X, use_splits=use_splits, **pred_params)
            pred = 1 / (1 + np.exp(-pred))
            preds.append(pred)

        self.preds = preds

        return np.mean(preds, axis=0)


class GridSearch(sklearn.base.BaseEstimator):
    """Custom analogue to sklearn's GridSearchCV with support for early stopping.
    Parameters
    ----------
    model : obj
    param_grid : dict of str to list
        Dictionary mapping each parameter to a list of candidate values to be searched.
    loss_fn : callable
        A function or callable loss object with signature `loss_fn(y_pred, y_true)`.
    Attributes
    ----------
    results_ : list of dict
        List of dictionaries recording fold, parameters, and loss from the grid search.
    results_df_ : pd.DataFrame
        A dataframe wrapper around `self.results_`.
    """

    def __init__(self, model, param_grid, loss_fn):
        self.model = model
        self.param_grid = param_grid
        self.loss_fn = loss_fn

    def fit(self, X, y=None, splits=None, **fit_params):

        """Split data into folds and train/evaluate parameter combinations on each fold.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training features.
        y : array-like, shape (n_samples, n_labels), optional
            The training labels, if required.
        splits : list of tuple
            The fold splits to  be used in training, in the form of a list of
            `(idx_train, idx_test)` tuples.
        **fit_params
            Additional fit parameters to pass to the model.
        Returns
        -------
        self : obj
            Returns the estimator itself.
        Notes
        -----
        This estimator does not save the trained models or implement a `best_model_`
        attribute or `predict` method. It is intended only for obtaining optimal
        parameter combos, most easily accessed in the `results_df_` attribtue.
        """

        loss_fn = (
            self.loss_fn if isinstance(self.loss_fn, dict) else {"loss": self.loss_fn}
        )

        # create parameter grid (result is list of parameter dicts)
        pg = sklearn.model_selection.ParameterGrid(self.param_grid)

        def to_tensor(a):
            return torch.tensor(a, dtype=torch.float32).to(self.model.device)

        # we'll record results for every param combo and fold
        self.results_ = []
        for params in pg:
            model = sklearn.base.clone(self.model).set_params(**params)
            # use our custom CV implementation to pass oos fold as val set
            cv = CVModel(model)
            cv.fit(X=X, y=y, splits=splits, **fit_params)
            # get oof error for every fold
            for i, m in enumerate(cv.models_):
                oof_indices = splits[i][1]
                y_pred = to_tensor(m.predict(_subset_rows(X, oof_indices)))
                y_true = to_tensor(_subset_rows(y, oof_indices))
                loss = {name: f(y_pred, y_true).item() for name, f in loss_fn.items()}
                # extract the name for network class so output is more readable
                params_copy = params.copy()
                if "net_obj" in params:
                    params_copy["net_obj"] = params_copy["net_obj"].name
                self.results_.append({"fold": i, **params_copy, **loss})

        # add dataframe version to easily read results
        self.results_df_ = pd.DataFrame(self.results_)

        return self


class OVRModel(sklearn.base.BaseEstimator):
    """Custom analogue to sklearn's OneVsRestClassifier with support for fit params.
    This is a very simple implementation that optionally paralellizes across outcomes
    and supports use of fit parameters. Prediction is not parallelized.
    Parameters
    ----------
    model : obj
        The estimator to be used for predicting each outcome. Currently there is no
        support for using different models for different outcomes.
    n_jobs : int, optional
        Number of joblib jobs to use in parallelization. The usual joblib rules apply,
        i.e. None -> 1 job, -1 -> all available.
    Attributes
    ----------
    models_ : list of obj
        The trained single-outcome models, which are used in prediction.
    """

    def __init__(self, model=None, n_jobs=None):
        self.model = model
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        """Fit a separate model for each outcome in y."""
        def parse_eval(fit_params, i):
            params = dict(fit_params)
            if "eval_set" in params:
                params["eval_set"] = [
                    (X_val, _subset_cols(y_val, i))
                    for (X_val, y_val) in params["eval_set"]
                ]
            return params

        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary)(
                self.model, X, _subset_cols(y, i), **parse_eval(fit_params, i)
            )
            for i in range(y.shape[1])
        )

    def predict(self, X):
        """Predict from each separate model and recombine."""

        y_pred = []

        for m in self.models_:
            # coerce to 1 dimension for models that return (n,1)
            p = m.predict(X).squeeze()
            y_pred.append(p)

        y_pred = np.array(y_pred).T

        return y_pred

    def predict_proba(self, X):
        """Get predicted probability from each separate model and recombine."""

        y_pred = []

        for m in self.models_:
            p = m.predict_proba(X)
            # take only p(1) when predict_proba returns 2 outcomes (sklearn, LGBM)
            if p.shape[1] == 2:
                p = p[:, 1]
            else:
                p = p.squeeze()
            y_pred.append(p)

        y_pred = np.array(y_pred).T

        return y_pred
