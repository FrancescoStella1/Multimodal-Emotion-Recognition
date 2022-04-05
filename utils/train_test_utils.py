import json, os, sys
import numpy as np
from enum import Enum

import GPUtil
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score, RocCurveDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

from skopt import BayesSearchCV
from skopt.space import Categorical, Real

from xgboost import plot_tree, XGBClassifier, XGBRFClassifier

from warnings import simplefilter

import main
from utils import load_store

simplefilter(action='ignore', category=FutureWarning)

DIMENSIONS = ["valence", "arousal"]  #, "dominance", "liking"]
MODELS_DIR = 'models'
RANDOM_STATE = 42
SAVE_PARAMS = False


class EMOTIONAL_STATE(Enum):
    """
    Enum containing the mapping for the valence and arousal ratings
    into two classes. This generates 9 discrete emotions in the Valence-Arousal plane.
    """
    LOW = (1, 3)
    MEDIUM = (3, 6)
    HIGH = (6, 9)


def apply_preprocessing(data, preprocessing, split_ratio=0.9):
    r"""
    Normalizes or standardizes the data given as input.

    Parameters
    ----------
    data: array_like
        Input data to be processed, it should have shape (samples, features).
    preprocessing: str
        Defines the type of preprocessing. It can be 'normalization' or 'standardization'.
    split_ratio: float
        Ratio of the data on which to fit the StandardScaler for standardization (default: 0.9).

    Returns
    -------
    data: array_like
        Processed data.
    """
    try:
        if preprocessing.lower() == 'normalization':
            data = normalize(data, norm='l1', axis=1)
        elif preprocessing.lower() == 'standardization':
            sc = StandardScaler()
            split = int(data.shape[0]*split_ratio)
            sc.fit(data[:split, :])
            data = sc.transform(data)
        return data

    except Exception as e:
        print("Error while preprocessing data - {}".format(repr(e)), file=sys.stderr)


def bayes_opt(model, params, data, labels, cv, optimizer_name, n_iter=32, scoring='accuracy', verbosity=0):
    r"""
    Performs Bayesian Optimization on hyperparameter space with the given model.

    Parameters
    ----------
    model: estimator object
        Model used for performing Hyperparameter Bayesian Optimization.
    params: dict
        Parameter space defined using scikit-optimize Dimension instances.
    data: array_like
        Training data.
    labels: array_like
        Ground truth labels.
    cv: cross validator object
        Cross validator object used to perform some kind of Cross Validation.
    scoring: str
        Score method to use for evaluation (default: 'accuracy').
    optimizer_name: str
        Filename used to save the optimizer (must end with .pkl extension) [currently not supported].
    n_iter: int
        Number of iterations/samplings to perform (default: 32).
    verbosity: int
        Verbosity level, useful for debugging (default: 0).

    Returns
    -------

    """
    search = BayesSearchCV(model, params, n_iter=n_iter, n_jobs=8, n_points=8, pre_dispatch='2*n_jobs', cv=cv,
                           return_train_score=True, scoring=scoring, verbose=verbosity)
    search.fit(data, labels)
    model = search.best_estimator_
    print("\n\n Best achieved score on left-out data: ", search.best_score_, "\n\n")
    return model


def labels_to_classes(labels, three_classes=False):
    r"""
    Converts real labels into integer labels representing classes.
    
    Parameters
    ----------
    labels: array_like
        Labels to convert.
    three_classes: bool
        Classify labels into three classes (default: False).

    Returns
    -------
    classes: array_like
        Array of the converted labels into the corresponding classes.
    """
    low = EMOTIONAL_STATE.LOW.value
    med = EMOTIONAL_STATE.MEDIUM.value
    new_labels = None
    if len(labels.shape) > 1:
        for dim in range(labels.shape[1]):
            if three_classes:
                classes = np.array(list(map(lambda x: 0 if (x <= low[1])
                                            else 1 if (x > med[0] and x <= med[1])
                                            else 2, labels[:, dim])))
            else:
                classes = np.where(labels[:, dim] <= 4.5, 0, 1)
            if new_labels is None:
                new_labels = classes
            else:
                new_labels = np.row_stack((new_labels, classes))
        return new_labels.T
    else:
        if three_classes:
            classes = np.array(list(map(lambda x: 0 if (x <= low[1])
                                        else 1 if (x > med[0] and x <= med[1])
                                        else 2, labels)))
        else:
            classes = np.where(labels <= 4.5, 0, 1)
        return classes


def remove_redundant_features(data, tolerance=0.15):
    r"""
    Removes redundant features from data.

    Parameters
    ----------
    data: array_like
        Input data with shape (samples, features).
    tolerance: float
        Tolerance value used to compute threshold for correlation (default: 0.15).

    Returns
    -------
    data: array_like
        Data without redundant features.
    """
    print("\n-> Removing redundant features..")
    print("Data shape: ", data.shape)
    corr_mat = np.corrcoef(data, rowvar=False)
    # plot.plot_corr_matrix(corr_mat)
    p = np.argwhere(np.triu(np.isclose(corr_mat, 1, atol=tolerance), 1))
    data = np.delete(data, p[:, 1], axis=1)
    print("New data shape: ", data.shape)
    return data


def svm_eeg_training(folds, task, data_preprocessing=None, scoring='accuracy', split_ratio=0.9, verbosity=0):
    r"""
    Trains an SVM on EEG data.

    Parameters
    ----------
    folds: int
        Number of folds used in K-fold cross validation.
    task: str
        Utility variable used internally for managing files. It refers to EEG_TASK in main, automatically computed.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    scoring: str
        Score method to use in Hyperparameter optimization (default: 'accuracy').
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Verbosity level (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related SVMs as values.
    """
    models = {}
    params = {
        'C': Real(1e-5, 50, 'uniform'),
        'gamma': Categorical(['scale']),
        'kernel': Categorical(['rbf']),
        'random_state': Categorical([RANDOM_STATE]),
        'tol': Real(1e-5, 1e-1, 'uniform')
    }
    cv = RepeatedStratifiedKFold(n_repeats=3, n_splits=folds, random_state=RANDOM_STATE)
    for dimension in DIMENSIONS:
        modelname = f"eeg_svm_{dimension}.pkl"
        modelpath = os.path.join(MODELS_DIR, modelname)
        if os.path.exists(modelpath):
            with open(modelpath, 'rb') as f:
                model = pickle.load(f)
                models[dimension] = model
            continue
        filename = f"{task}_{dimension}.npz"
        # Load data and labels for each dimension
        data, labels = load_store.load_fe_eeg(main.EEG_SAVE_PATH, filename)
        if data_preprocessing is not None:
            data = apply_preprocessing(data, data_preprocessing)
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)
        model = bayes_opt(SVC(), params, X_train, Y_train, cv, f"optimizer_eeg_{dimension}.pkl", n_iter=96,
                          scoring=scoring, verbosity=verbosity)
        with open(modelpath, 'wb') as f:
            pickle.dump(model, f)
        models[dimension] = model
    return models


def svm_ensemble_training(folds, model_type, data_preprocessing=None, scoring='accuracy', split_ratio=0.9,
                          verbosity=0):
    r"""
    Trains SVM for decision-level fusion, using predictions from EEG and Facial Features.

    Parameters
    ----------
    folds: int
        Number of folds used in K-fold cross validation.
    model_type: str
        Utility variable for managing files, it refers to ENSEMBLE_MODEL_TYPE in main, automatically computed.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    scoring: str
        Score method to use in Hyperparameter optimization (default: 'accuracy').
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Verbosity level (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related SVMs as values.
    """
    models = {}
    params = {
        'C': Real(1e-5, 50, 'uniform'),
        'gamma': Categorical(['scale', 'auto']),
        'kernel': Categorical(['rbf']),
        'random_state': Categorical([RANDOM_STATE]),
        'tol': Real(1e-5, 1e-1, 'uniform')
    }
    cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=folds, random_state=RANDOM_STATE)
    for dimension in DIMENSIONS:
        eeg, ff, labels = load_store.load_ensemble_data(main.ENSEMBLE_SAVE_PATH, f"ensemble_{dimension}.npz")
        data = np.column_stack((eeg, ff))
        if data_preprocessing is not None:
            data = apply_preprocessing(data, data_preprocessing, split_ratio=split_ratio)
        filename = f"ensemble_{model_type}_{dimension}.pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        model = load_store.load_ensemble_model(MODELS_DIR, filename)
        if model is not None:
            models[dimension] = model
            continue
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)
        model = bayes_opt(SVC(), params, X_train, Y_train, cv, f"optimizer_ensemble_{dimension}.pkl", n_iter=96,
                          scoring=scoring, verbosity=verbosity)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        models[dimension] = model
    return models


def svm_ff_training(folds, data_preprocessing=None, scoring='accuracy', split_ratio=0.9,
                    verbosity=0):
    r"""
    Trains an SVM on Facial Features data.

    Parameters
    ----------
    folds: int
        Number of folds used in K-fold cross validation.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    scoring: str
        Score method to use in Hyperparameter optimization (default: 'accuracy').
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Verbosity level (default: 0).
    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and SVMs as values.
    """
    models = {}
    params = {
        'C': Real(1e-5, 50, 'uniform'),
        'gamma': Categorical(['scale', 'auto']),
        'kernel': Categorical(['rbf']),
        'random_state': Categorical([RANDOM_STATE]),
        'tol': Real(1e-5, 1e-1, 'uniform')
    }
    cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=folds, random_state=RANDOM_STATE)
    ff, labels = load_store.load_ff_data(main.FF_SAVE_PATH, f"{main.FF_MODEL_TYPE}.npz", return_labels=True)
    if data_preprocessing is not None:
        ff = apply_preprocessing(ff, data_preprocessing)
    for dimension in DIMENSIONS:
        filename = f"ff_svm_{dimension}.pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        model = load_store.load_ff_model(MODELS_DIR, filename)
        if model is not None:
            models[dimension] = model
            continue
        idx = DIMENSIONS.index(dimension)
        X_train, X_test, Y_train, Y_test = train_test_split(ff, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)
        model = bayes_opt(SVC(), params, X_train, Y_train[:, idx], cv, f"optimizer_ff_{dimension}.pkl",
                          scoring=scoring, verbosity=verbosity)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        models[dimension] = model
    return models


def xgb_dt_ensemble_training(folds, model_type, data_preprocessing=None, scoring='accuracy', split_ratio=0.9,
                             verbosity=0):
    r"""
    Trains a Decision Tree for decision-level fusion.

    Parameters
    ----------
    folds: int
        Number of folds used in K-fold cross validation.
    model_type: str
        Utility variable used for managing files. It refers to ENSEMBLE_MODEL_TYPE in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    scoring: str
        Score method to use in Hyperparameter optimization (default: 'accuracy').
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Verbosity level (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related Random forests as values.
    """
    models = {}
    params = {
        'eval_metric': Categorical(['error']),
        'learning_rate': Categorical([1e-2]),
        'num_parallel_tree': Categorical([10]),
        'n_jobs': Categorical([4]),
        'objective': Categorical(['binary:logistic']),
        'random_state': Categorical([RANDOM_STATE]),
        'reg_alpha': Real(1e-5, 50, 'uniform'),
        'reg_lambda': Real(1e-5, 50, 'uniform'),
        'use_label_encoder': Categorical([False]),
        'tree_method': Categorical(['hist'])
    }
    cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=folds, random_state=RANDOM_STATE)
    for dimension in DIMENSIONS:
        eeg, ff, labels = load_store.load_ensemble_data(main.ENSEMBLE_SAVE_PATH, f"ensemble_{dimension}.npz")
        data = np.column_stack((eeg, ff))
        if data_preprocessing is not None:
            data = apply_preprocessing(data, data_preprocessing, split_ratio=split_ratio)
        filename = f"ensemble_{model_type}_{dimension}"
        model = load_store.load_ensemble_model(MODELS_DIR, filename)
        if model is not None:
            models[dimension] = model
            continue
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)
        model = bayes_opt(XGBClassifier(), params, X_train, Y_train, cv, f"ensemble_{dimension}.pkl",
                          n_iter=64, scoring=scoring, verbosity=verbosity)
        model.save_model(os.path.join(MODELS_DIR, f"ensemble_{model_type}_{dimension}.json"))
        models[dimension] = model
    return models


def xgb_rf_classification(folds, data_type=None, task='rf_clf', data_preprocessing=None, model_type='xgb_rf_clf',
                          scoring='accuracy', split_ratio=0.9, verbosity=0):
    r"""
    Trains a Random Forest classifier on EEG or Facial Features data.

    Parameters
    ----------
    folds: int
        Number of folds used in K-fold cross validation.
    data_type: str
        Used to train EEF or Facial Features, it should be 'eeg' or 'ff' (default: None).
    task: str
        Utility variable used internally for managing files (default: 'rf_clf').
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    model_type: str
        Utility variable describing the type of model to train (default: 'xgb_rf_clf').
    scoring: str
        Score method to use in Hyperparameter optimization (default: 'accuracy').
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Verbosity level (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related Random forests as values.
    """
    models = {}
    params = {
        'eval_metric': Categorical(['error']),
        'grow_policy': Categorical(['depthwise']),      #  splits at nodes closest to the root
        'learning_rate': Categorical([1]),              #  must be set to 1 in Random Forests
        'n_estimators': Categorical([200]),
        'n_jobs': Categorical([4]),
        'objective': Categorical(['binary:logistic']),
        'random_state': Categorical([RANDOM_STATE]),
        'reg_alpha': Real(1e-3, 50, 'uniform'),
        'reg_lambda': Real(1e-3, 50, 'uniform'),
        'subsample': Real(0.2, 0.9, 'uniform'),
        'use_label_encoder': Categorical([False])
    }

    if data_type == 'eeg':
        params['colsample_bynode'] = Real(0.2, 0.9, 'uniform')

    if len(GPUtil.getAvailable()) > 0:
        params['gpu_id'] = Categorical(['0'])
        params['predictor'] = Categorical(['gpu_predictor'])
        params['tree_method'] = Categorical(['gpu_hist'])

    else:
        params['n_threads'] = Categorical([-1])
        params['tree_method'] = Categorical(['hist'])

    cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=folds, random_state=RANDOM_STATE)
    for dimension in DIMENSIONS:
        filename = f"{data_type}_xgb_rf_clf_{dimension}"
        model = load_store.load_eeg_model(MODELS_DIR, filename)
        if model is not None:
            models[dimension] = model
            continue
        filename = f"{task}_{dimension}.npz"
        # Load data and labels for RF classifier and for each dimension
        if data_type == 'eeg':
            data, labels = load_store.load_fe_eeg(main.EEG_SAVE_PATH, filename)
        else:
            idx = DIMENSIONS.index(dimension)
            data, labels = load_store.load_ff_data(main.FF_SAVE_PATH, f"{model_type}.npz", return_labels=True)
            labels = labels[:, idx]
        if data_preprocessing is not None:
            data = apply_preprocessing(data, data_preprocessing)
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)
        model = bayes_opt(XGBRFClassifier(), params, X_train, Y_train, cv, f"optimizer_{data_type}_{dimension}.pkl",
                          n_iter=64, scoring=scoring, verbosity=verbosity)
        model.save_model(os.path.join(MODELS_DIR, f"{data_type}_xgb_rf_clf_{dimension}.json"))
        models[dimension] = model
    return models


def xgb_rf_ensemble_training(folds, model_type, data_preprocessing=None, scoring='accuracy', split_ratio=0.9,
                             verbosity=0):
    r"""
    Trains a Random Forest for decision-level fusion.

    Parameters
    ----------
    folds: int
        Number of folds used in K-fold cross validation.
    model_type: str
        Utility variable used for managing files. It refers to ENSEMBLE_MODEL_TYPE in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    scoring: str
        Score method to use in Hyperparameter optimization (default: 'accuracy').
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Verbosity level (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related Random forests as values.
    """
    models = {}
    params = {
        'colsample_bynode': Real(0.2, 0.9, 'uniform'),
        'eval_metric': Categorical(['error']),
        'grow_policy': Categorical(['depthwise']),
        'learning_rate': Categorical([1]),
        'n_estimators': Categorical([100]),
        'n_jobs': Categorical([4]),
        'objective': Categorical(['binary:logistic']),
        'random_state': Categorical([RANDOM_STATE]),
        'reg_alpha': Real(1e-5, 50, 'uniform'),
        'reg_lambda': Real(1e-5, 50, 'uniform'),
        'subsample': Real(0.2, 0.9, 'uniform'),
        'use_label_encoder': Categorical([False])
    }
    if len(GPUtil.getAvailable()) > 0:
        params['gpu_id'] = Categorical(['0'])
        params['predictor'] = Categorical(['gpu_predictor'])
        params['tree_method'] = Categorical(['gpu_hist'])

    else:
        params['tree_method'] = Categorical(['hist'])
        params['n_threads'] = Categorical([-1])

    cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=folds, random_state=RANDOM_STATE)
    for dimension in DIMENSIONS:
        eeg, ff, labels = load_store.load_ensemble_data(main.ENSEMBLE_SAVE_PATH, f"ensemble_{dimension}.npz")
        data = np.column_stack((eeg, ff))
        if data_preprocessing is not None:
            data = apply_preprocessing(data, data_preprocessing, split_ratio=split_ratio)
        filename = f"ensemble_{model_type}_{dimension}"
        model = load_store.load_ensemble_model(MODELS_DIR, filename)
        if model is not None:
            models[dimension] = model
            continue
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)
        model = bayes_opt(XGBRFClassifier(), params, X_train, Y_train, cv, f"ensemble_{dimension}.pkl",
                          scoring=scoring, verbosity=verbosity)
        model.save_model(os.path.join(MODELS_DIR, f"ensemble_{model_type}_{dimension}.json"))
        models[dimension] = model
    return models


def test_eeg_models(model_type, task, data_preprocessing=None, split_ratio=0.9):
    r"""
    Tests EEG models.

    Parameters
    ----------
    model_type: str
        Indicates the models to test. It refers to EEG_MODEL_TYPE in main.
    task: str
        Utility variable used internally. It refers to EEG_TASK in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    split_ratio: float
         Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    """
    try:
        model = None
        for dimension in DIMENSIONS:
            if model_type == 'xgb_rf_clf':
                model = XGBRFClassifier()
            elif model_type == 'svm':
                modelname = f"eeg_svm_{dimension}.pkl"
                modelpath = os.path.join(MODELS_DIR, modelname)
                with open(modelpath, 'rb') as f:
                    model = pickle.load(f)
            else:
                return model
            filename = f"{task}_{dimension}.npz"
            eeg, labels = load_store.load_fe_eeg(main.EEG_SAVE_PATH, filename)
            if model_type != "svm":
                model.load_model(os.path.join(MODELS_DIR, f"eeg_{model_type}_{dimension}.json"))
            if data_preprocessing is not None:
                eeg = apply_preprocessing(eeg, data_preprocessing)
            X_train, X_test, Y_train, Y_test = train_test_split(eeg, labels, train_size=split_ratio,
                                                                random_state=RANDOM_STATE)
            score = model.score(X_test, Y_test)
            pred = model.predict(X_test)
            f1 = f1_score(Y_test, pred)
            ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, normalize='true')
            print(f"Accuracy for {dimension} using {model_type}: %.3f " % score)
            print(f"F1 score for {dimension} using {model_type}: %.3f " % f1)
            if SAVE_PARAMS:
                params = json.dumps(model.get_params(), indent=4, sort_keys=True, separators=(',', ': '))
                with open(f'eeg_{model_type}_{dimension}_params.json', 'w') as f:
                    f.write(params)
            plt.show()
            RocCurveDisplay.from_estimator(model, X_test, Y_test)
            plt.show()
            # plot_tree(model, num_trees=0, rankdir='LR')
            # plt.savefig(f"tree_{dimension}.png", dpi=1600)

    except Exception as e:
        print("Error during testing of EEG models - {}".format(repr(e)), file=sys.stderr)


def test_ensemble(model_type, data_preprocessing=None, split_ratio=0.9):
    r"""
    Tests the model used for decision-level fusion.

    Parameters
    ----------
    model_type: str
        Indicates the models to test. It refers to ENSEMBLE_MODEL_TYPE in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    """
    for dimension in DIMENSIONS:
        filename = f"ensemble_{dimension}.npz"
        eeg, ff, labels = load_store.load_ensemble_data(main.ENSEMBLE_SAVE_PATH, filename)
        data = np.column_stack((eeg, ff))
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)
        model = load_store.load_ensemble_model(MODELS_DIR, f"ensemble_{model_type}_{dimension}")
        if model is None:
            continue
        pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, pred)
        f1 = f1_score(Y_test, pred)
        ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, normalize='true')
        print(f"Accuracy for {dimension} using ensemble: %.3f" % accuracy)
        print(f"F1 score for {dimension} using ensemble: %.3f" % f1)
        if SAVE_PARAMS:
            params = json.dumps(model.get_params(), indent=4, sort_keys=True, separators=(',', ': '))
            with open(f'ensemble_{model_type}_{dimension}_params.json', 'w') as f:
                f.write(params)
        plt.show()
        RocCurveDisplay.from_estimator(model, X_test, Y_test)
        plt.show()


def test_ff_models(model_type, data_preprocessing=None, split_ratio=0.9):
    r"""
    Tests Facial Features models.

    Parameters
    ----------
    model_type: str
        Indicates the models to test. It refers to FF_MODEL_TYPE in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    split_ratio: float
        Ratio of the entire dataset used for sampling training set, it's a value in [0,1] (default: 0.9).
    """
    try:
        ff, labels = load_store.load_ff_data(main.FF_SAVE_PATH, f"{model_type}.npz", return_labels=True)
        if data_preprocessing is not None:
            ff = apply_preprocessing(ff, data_preprocessing)
        X_train, X_test, Y_train, Y_test = train_test_split(ff, labels, train_size=split_ratio,
                                                            random_state=RANDOM_STATE)

        for dimension in DIMENSIONS:
            filename = f"{model_type}_{dimension}"
            if model_type == 'svm':
                model = load_store.load_ff_model(MODELS_DIR, f"ff_{filename}.pkl")
            if model_type == 'xgb_rf_clf':
                model = load_store.load_ff_model(MODELS_DIR, f"ff_{filename}.json")
            else:
                return None
            idx = DIMENSIONS.index(dimension)
            pred = model.predict(X_test)
            score = accuracy_score(Y_test[:, idx], pred)
            f1 = f1_score(Y_test[:, idx], pred)
            ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test[:, idx], normalize='true')
            print(f"Accuracy for {dimension} using {model_type}: %.3f" % score)
            print(f"F1 score for {dimension} using {model_type}: %.3f" % f1)
            # print(model.get_params())
            if SAVE_PARAMS:
                params = json.dumps(model.get_params(), indent=4, sort_keys=True, separators=(',', ': '))
                with open(f'ff_ensemble_{model_type}_{dimension}_params.json', 'w') as f:
                    f.write(params)
            plt.show()
            RocCurveDisplay.from_estimator(model, X_test, Y_test[:, idx])
            plt.show()

    except Exception as e:
        print("Error during testing of Facial Features models - {}".format(repr(e)), file=sys.stderr)


def train_eeg(model_type, task, data_preprocessing=None, folds=3, split_ratio=0.9, verbosity=0):
    r"""
    Trains the models used for EEG related predictions.

    Parameters
    ----------
    model_type: str
        Indicates the models to train. It refers to EEG_MODEL_TYPE in main.
    task: str
        Utility variable used internally. It refers to EEG_TASK in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    folds: int
        Number of folds to use in K-fold cross validation (default: 3).
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Level of verbosity for debugging purposes (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related models as values.
    """
    models = None
    if model_type.lower() == "svm":
        models = svm_eeg_training(folds, task, data_preprocessing=data_preprocessing, split_ratio=split_ratio,
                                  verbosity=verbosity)

    elif model_type.lower() == "xgb_rf_clf":
        models = xgb_rf_classification(folds, data_type='eeg', task=task, data_preprocessing=data_preprocessing,
                                       split_ratio=split_ratio, verbosity=verbosity)

    return models


def train_ensemble(eeg_model_type, ff_model_type, model_type, task, data_preprocessing=None, folds=3, split_ratio=0.9,
                   verbosity=0):
    r"""
    Trains models for performing decision-level fusion. If EEG and FF models are not already trained, it first trains
    them.

    Parameters
    ----------
    eeg_model_type: str
        Indicates the EEG models to train. It refers to EEG_MODEL_TYPE in main.
    ff_model_type: str
        Indicates the FF models to train. It refers to FF_MODEL_TYPE in main.
    model_type: str
        Indicates the model to train for decision-level fusion. It refers to ENSEMBLE_MODEL_TYPE in main.
    task: str
        Utility variable used internally. It refers to EEG_TASK in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    folds: int
        Number of folds to use in K-fold cross validation (default: 3).
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Level of verbosity for debugging purposes (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related models as values.
    """
    for dimension in DIMENSIONS:
        filename = f"ensemble_{dimension}.npz"
        filepath = os.path.join(main.ENSEMBLE_SAVE_PATH, filename)
        if not os.path.exists(filepath):
            eeg_models = train_eeg(eeg_model_type, task, data_preprocessing=data_preprocessing, folds=folds,
                                   split_ratio=split_ratio, verbosity=verbosity)
            ff_models = train_ff(ff_model_type, data_preprocessing=data_preprocessing, folds=folds,
                                 split_ratio=split_ratio, verbosity=verbosity)
            eeg_model = eeg_models[dimension]
            ff_model = ff_models[dimension]
            eeg_pred, ff_pred, labels = merge_predictions(eeg_model, ff_model, dimension, task)
            load_store.save_ensemble_data(main.ENSEMBLE_SAVE_PATH, f"ensemble_{dimension}.npz", eeg_pred, ff_pred,
                                          labels)
    models = None
    if model_type == 'svm':
        models = svm_ensemble_training(folds, model_type, data_preprocessing=data_preprocessing,
                                       split_ratio=split_ratio, verbosity=verbosity)
    elif model_type == 'xgb_rf_clf':
        models = xgb_rf_ensemble_training(folds, model_type, data_preprocessing=data_preprocessing,
                                          split_ratio=split_ratio, verbosity=verbosity)
    elif model_type == 'xgb_dt_clf':
        models = xgb_dt_ensemble_training(folds, model_type, data_preprocessing=data_preprocessing,
                                          split_ratio=split_ratio, verbosity=verbosity)
    return models


def train_ff(model_type, data_preprocessing=None, folds=3, split_ratio=0.9, verbosity=0):
    r"""
    Trains the models used for Facial Features (FF) predictions.

    Parameters
    ----------
    model_type: str
        Indicates the models to train. It refers to FF_MODEL_TYPE in main.
    data_preprocessing: str
        It can be 'normalization' or 'standardization' if needed (default: None).
    folds: int
        Number of folds to use in K-fold cross validation (default: 3).
    split_ratio: float
        Ratio of the entire dataset used for sampling training set. It's a value in [0,1] (default: 0.9).
    verbosity: int
        Level of verbosity for debugging purposes (default: 0).

    Returns
    -------
    models: dict
        Dict containing dimensions as keys (e.g. Valence, Arousal) and the related models as values.
    """
    models = None
    if model_type == 'svm':
        models = svm_ff_training(folds, data_preprocessing=data_preprocessing, scoring='accuracy',
                                 split_ratio=split_ratio, verbosity=verbosity)
    elif model_type == 'xgb_rf_clf':
        models = xgb_rf_classification(folds, data_type='ff', data_preprocessing=data_preprocessing,
                                       model_type=model_type, split_ratio=split_ratio, verbosity=verbosity)
    return models


def merge_predictions(eeg_model, ff_model, dimension, task):
    r"""
    Merges the predictions from EEG and FF models for each dimension (Valence, Arousal, etc.), each subject and
    each trial.

    Parameters
    ----------
    eeg_model: object
        Model used for EEG predictions.
    ff_model: object
        Model used for FF predictions.
    dimension: str
        Dimension associated to the predictions (e.g. Valence, Arousal).
    task: str
        Utility variable used internally. It refers to EEG_TASK in main.

    Returns
    -------
    eeg_predictions: array_like
        EEG predictions for the given dimension.
    ff_predictions: array_like
        FF predictions for the given dimension.
    final_labels: array_like
        Ground truth labels.
    """
    print(f"\nMerging EEG and FF models predictions for {dimension}..\n\n")
    entries = os.listdir(main.FF_SAVE_PATH)
    eeg_filename = f"{task}_{dimension}.npz"
    eeg_data, labels = load_store.load_fe_eeg(main.EEG_SAVE_PATH, eeg_filename)
    trial_cnt = -1
    ff_predictions = None
    eeg_predictions = None
    final_labels = None

    for entry in entries:
        if not entry.startswith('s') or 'svm' in entry:
            continue
        ff = load_store.load_ff_data(main.FF_SAVE_PATH, entry)
        for trial in range(ff.shape[0]):
            trial_cnt += 1
            trial_pred = None
            # Predict each of the 120 samples from each trial
            for sample in range(ff.shape[1]):
                if not np.any(np.isnan(ff[trial, sample, :])):
                    smp_pred = ff_model.predict(ff[trial, sample, :].reshape(1, -1))
                    if trial_pred is None:
                        trial_pred = smp_pred
                    else:
                        trial_pred = np.concatenate((trial_pred, smp_pred))
            # Trial is all Nan, i.e. it is missing from video recordings
            if trial_pred is None:
                continue
            # Get the most frequent prediction for each trial, or 0
            bc = np.bincount(trial_pred)
            trial_pred = np.array([np.argmax(bc)])
            eeg_pred = eeg_model.predict(eeg_data[trial_cnt, :].reshape(1, -1))
            label = np.array([labels[trial_cnt]])
            if ff_predictions is None:
                ff_predictions = trial_pred
            else:
                ff_predictions = np.concatenate((ff_predictions, trial_pred))
            if eeg_predictions is None:
                eeg_predictions = eeg_pred
            else:
                eeg_predictions = np.concatenate((eeg_predictions, eeg_pred))
            if final_labels is None:
                final_labels = label
            else:
                final_labels = np.concatenate((final_labels, label))
    return eeg_predictions, ff_predictions, final_labels
