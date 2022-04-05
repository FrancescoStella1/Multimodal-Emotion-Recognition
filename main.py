import numpy as np
import os, sys
import argparse, itertools

from utils import landmark_detection, load_store, prepare_eeg_data, train_test_utils

AVOID_PLOTS = True
FS = 128                                            # Sampling Frequency from preprocessed DEAP dataset
LANDMARKS_DATA_PATH = "data"
USE_LBFMODEL = False
VIDEO_DATA_PATH = os.path.join("..", "..", "deap")

EEG_DATA_PATH = os.path.join("..", "..", "deap", "eeg_data")
EEG_FEATURE_TYPE = 'rpsd'
EEG_MODEL_TYPE = "svm"
EEG_PREPROC = "standardization"
EEG_SAVE_PATH = os.path.join("preprocessed", "eeg")  # Path of the preprocessed files
EEG_SPLIT = 0.8                                      # Ratio of the dataset to perform train/test split
EEG_TASK = None

ENSEMBLE_MODEL_TYPE = "svm"
ENSEMBLE_PREPROC = None
ENSEMBLE_SAVE_PATH = os.path.join("preprocessed", "ensemble")
ENSEMBLE_SPLIT = 0.8

FF_MODEL_TYPE = "xgb_rf_clf"
FF_PREPROC = None
FF_SAVE_PATH = os.path.join("preprocessed", "facial_features")
FF_SPLIT = 0.8


def create_eeg_files():
    r"""
    Reads and preprocesses EEG data from .dat files, then extracts features and removes highly correlated features.
    The resulting data and labels are stored in one file for each considered dimension (e.g. Valence, Arousal, etc.).
    """
    try:
        global AVOID_PLOTS, EEG_TASK
        if not os.path.exists(EEG_SAVE_PATH):
            os.makedirs(EEG_SAVE_PATH)
        entries = set(os.listdir(EEG_DATA_PATH))
        entries = set(filter(lambda x: x[-4:] == ".dat", entries))
        already_done = set(os.listdir(EEG_SAVE_PATH))

        while (len(entries) > 0):
            entry = entries.pop()
            subj = entry[:entry.index(".")]
            if "{}.npz".format(subj) in already_done:
                continue
            print("Processing {} ...\n".format(entry))
            freqs, bands, labels = prepare_eeg_data.get(os.path.join(EEG_DATA_PATH, entry), avoid_plot=AVOID_PLOTS,
                                                        feature_type=EEG_FEATURE_TYPE, sampling_freq=FS)
            AVOID_PLOTS = True
            load_store.save_eeg_data(EEG_SAVE_PATH, subj, freqs, bands, labels)

        eeg, labels = load_store.load_all_eeg_data(EEG_SAVE_PATH)
        if 'svm' in EEG_MODEL_TYPE or 'clf' in EEG_MODEL_TYPE:
            labels = train_test_utils.labels_to_classes(labels)
            task = 'clf'
        else:
            task = 'reg'
        if 'rf' in EEG_MODEL_TYPE:
            task = 'rf_'+task
        EEG_TASK = task
        for dimension in train_test_utils.DIMENSIONS:
            filename = f"{task}_{dimension}.npz"
            filepath = os.path.join(EEG_SAVE_PATH, filename)
            if not os.path.exists(filepath):
                idx = train_test_utils.DIMENSIONS.index(dimension)
                reduced_eeg = train_test_utils.remove_redundant_features(eeg, tolerance=0.15)
                '''
                if task == 'clf':
                    reduced_eeg = train_test_utils.apply_rfecv(reduced_eeg, labels[:, idx], 3, 'accuracy', 30,
                                                               preprocessing=EEG_PREPROC)
                '''
                load_store.save_fe_eeg(EEG_SAVE_PATH, filename, reduced_eeg, labels[:, idx])

    except Exception as e:
        print("Exception while creating eeg files: ", repr(e), file=sys.stderr)


def merge_ff_data():
    r"""
    Reads facial features from .npz files and stores them into a single .npz file.
    """
    filename = f"{FF_MODEL_TYPE}.npz"
    if os.path.exists(os.path.join(FF_SAVE_PATH, filename)):
        return
    entries = os.listdir(FF_SAVE_PATH)
    ff_data = None
    labels_data = None
    for entry in entries:
        if not entry.startswith('s') or 'svm' in entry:
            continue
        ff = load_store.load_ff_data(FF_SAVE_PATH, entry)
        _, _, labels = load_store.load_eeg_data(EEG_SAVE_PATH, entry)
        if "svm" in FF_MODEL_TYPE or "clf" in FF_MODEL_TYPE:
            labels = train_test_utils.labels_to_classes(labels)
        labels = np.repeat(labels, ff.shape[1], axis=0)
        if labels_data is None:
            labels_data = labels
        else:
            labels_data = np.row_stack((labels_data, labels))
        for tr in range(ff.shape[0]):
            if ff_data is None:
                ff_data = ff[tr, :, :]
            else:
                ff_data = np.row_stack((ff_data, ff[tr, :, :]))
    idxs = np.where(np.any(np.isnan(ff_data), axis=1) == False)[0]
    ff_data = ff_data[idxs, :]
    labels_data = labels_data[idxs, :]
    load_store.save_ff_data(FF_SAVE_PATH, filename, ff_data, labels_data)


'''
Each participant file contains two arrays: 
    - data[40x40x8064] (video_trial x channel x data(= 63 seconds at 128Hz)).
    - labels[40x4] (video_trial x label(= valence, arousal, dominance, liking))
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_classes', type=bool, default=False,
                        help='Checks distribution of the classes among all trials and subjects')
    parser.add_argument('--create_eeg_files', type=bool, default=False,
                        help='Applies feature extraction on eeg input data and stores the results into .npz files'
                             ' (default: False)')
    parser.add_argument('--create_ff_files', type=bool, default=False,
                        help='Applies facial features extraction on landmark files and stores the result into .npz '
                             'files (default: False)')
    parser.add_argument('--create_landmark_files', type=bool, default=False,
                        help='Applies landmarks extraction on video input data and stores the results into .npz files '
                             '(default: False)')
    parser.add_argument('--test_eeg', type=bool, default=False, help='Test EEG models (default: False)')
    parser.add_argument('--test_ensemble', type=bool, default=False, help='Test the ensemble of models at '
                                                                          'decision-level (default: False)')
    parser.add_argument('--test_ff', type=bool, default=False, help='Test Facial Feature models (default: False)')
    parser.add_argument('--train_eeg', type=bool, default=False, help='Train SVM for EEG features (default: False)')
    parser.add_argument('--train_ensemble', type=bool, default=False,
                        help='Train the decision-level model starting from EEG and Facial Features'
                             'models (default: False)')
    parser.add_argument('--train_ff', type=bool, default=False, help='Train SVM for facial features (default: False)')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("\nPlease, specify at least one option.\n\n", file=sys.stderr)
        print(parser.print_help())
        sys.exit(1)

    if args.check_classes:
        entries = set(os.listdir(EEG_SAVE_PATH))
        pv_cnt = 0
        nv_cnt = 0
        ha_cnt = 0
        la_cnt = 0
        _, labels = load_store.load_all_eeg_data(EEG_SAVE_PATH)
        split = int(labels.shape[0] * EEG_SPLIT)
        np.random.shuffle(labels)
        labels = labels[:split, :2]
        for label in labels:
            if label[0] > 4.5:  # positive valence
                pv_cnt += 1
            else:
                nv_cnt += 1
            if label[1] > 4.5:  # high arousal
                ha_cnt += 1
            else:
                la_cnt += 1

        print("\n[ Test set stats on class balance ]\n\n")
        print("Total # of positive valence labels: ", pv_cnt)
        print("Total # of negative valence labels: ", nv_cnt)
        print("Total # of high arousal labels: ", ha_cnt)
        print("Total # of low arousal labels: ", la_cnt)
        print("\nRatio high arousal/positive valence: %.2f" % (ha_cnt/pv_cnt))
        print("Ratio low arousal/negative valence: %.2f" % (la_cnt/nv_cnt))

    if args.create_eeg_files:
        create_eeg_files()

    if args.create_ff_files:
        entries = set(os.listdir(LANDMARKS_DATA_PATH))
        if len(entries) == 0: print("\nError: cannot find landmarks files in '{}'. Please, first generate them." \
                                    .format(LANDMARKS_DATA_PATH), file=sys.stderr)
        default_shape = (40, 120, 7)  # (trials, frames per trial, features)
        features_array = np.array([])
        filename = None
        prev_subj = "01"

        for subj, trial in itertools.product(np.arange(1, 23, 1), np.arange(1, 41, 1)):
            if subj < 10: subj = "0{}".format(subj)
            if trial < 10: trial = "0{}".format(trial)
            filename = "s{}_landmarks_trial{}.npz".format(subj, trial)
            if not os.path.exists(os.path.join(LANDMARKS_DATA_PATH, filename)):
                print("File {} doesn't exist!".format(filename))
                continue
            facial_features = landmark_detection.get_features_mp(LANDMARKS_DATA_PATH, filename)

            # Check for missing landmarks/features and pad accordingly
            if facial_features.shape[0] < 120:
                diff = default_shape[1] - facial_features.shape[0]
                padding = np.full((diff, default_shape[2]), np.nan)
                facial_features = np.row_stack((facial_features, padding))

            facial_features = np.reshape(facial_features, (1, facial_features.shape[0], facial_features.shape[1]))

            # Store .npz file
            if prev_subj != subj:
                filename = "s{}.npz".format(prev_subj)
                # Check for missing trials and pad accordingly
                if features_array.shape[0] < 40:
                    diff = default_shape[0] - features_array.shape[0]
                    padding = np.full((diff, default_shape[1], default_shape[2]), np.nan)
                    features_array = np.row_stack((features_array, padding))
                load_store.save_ff_data(FF_SAVE_PATH, filename, features_array)
                print("Computing facial features of subject {}..".format(subj))
                features_array = np.array([])
                prev_subj = subj

            if len(features_array) == 0:
                features_array = facial_features
            else:
                features_array = np.row_stack((features_array, facial_features))

        # Store .npz file for last subject
        filename = "s22.npz"
        # Check for missing trials and pad accordingly
        if features_array.shape[0] < 40:
            diff = default_shape[0] - features_array.shape[0]
            padding = np.full((diff, default_shape[1], default_shape[2]), np.nan)
            features_array = np.row_stack((features_array, padding))
        load_store.save_ff_data(FF_SAVE_PATH, filename, features_array)

    if args.create_landmark_files:
        if USE_LBFMODEL:
            if not os.path.exists(FF_SAVE_PATH): os.makedirs(FF_SAVE_PATH)
            face_model, landmarks_model = landmark_detection.load_models()
            facial_features_batch = np.array([])

        else:
            if not os.path.exists("data"): os.makedirs("data")

        entries = set(os.listdir(VIDEO_DATA_PATH))
        entries = set(filter(lambda x: ".avi" in x, entries))
        try:
            while len(entries) > 0:
                entry = entries.pop()
                subj = entry[:3]
                starting_idx = entry.index("trial") + len("trial")
                trial_num = entry[starting_idx:entry.index(".")]

                # print("-> Processing video for subject {} and trial {}..".format(subj, trial_num))

                if (USE_LBFMODEL):
                    facial_features = landmark_detection.get_features(os.path.join(VIDEO_DATA_PATH, filename),
                                                                      face_model, landmarks_model)
                    print("Facial features shape for subject {} and trial {}: ".format(subj, trial_num),
                          facial_features.shape)
                    # assert(facial_features.shape == (60, 7))

                else:
                    filename = "{}_landmarks_trial{}.npz".format(subj, trial_num)
                    facial_landmarks = landmark_detection.get_landmarks_mp(os.path.join(VIDEO_DATA_PATH, entry), 25)
                    if (facial_landmarks is not None): load_store.save_landmark_data(LANDMARKS_DATA_PATH, filename,
                                                                                     facial_landmarks)

        except Exception as e:
            print("Exception while creating video files: ", repr(e), file=sys.stderr)
            sys.exit(1)

    if args.test_eeg:
        print("\n\n[ TEST EEG MODELS ]\n\n")
        create_eeg_files()
        train_test_utils.test_eeg_models(EEG_MODEL_TYPE, EEG_TASK, data_preprocessing=EEG_PREPROC, split_ratio=EEG_SPLIT)
        print("Done.")

    if args.test_ensemble:
        print("\n\n[ TEST ENSEMBLE OF MODELS ]\n\n")
        train_test_utils.test_ensemble(ENSEMBLE_MODEL_TYPE, split_ratio=EEG_SPLIT)

    if args.test_ff:
        print("\n\n[ TEST FACIAL FEATURE MODELS ]\n\n")
        merge_ff_data()
        train_test_utils.test_ff_models(FF_MODEL_TYPE, data_preprocessing=FF_PREPROC, split_ratio=FF_SPLIT)
        print("Done.")

    if args.train_eeg:
        print('\n\n[ TRAIN EEG MODEL ]\n\n')
        create_eeg_files()
        train_test_utils.train_eeg(EEG_MODEL_TYPE, EEG_TASK, data_preprocessing=EEG_PREPROC, folds=5,
                                   split_ratio=EEG_SPLIT, verbosity=2)
        print("Done.")

    if args.train_ensemble:
        print("\n\n[ TRAIN ENSEMBLE OF MODELS ]")
        create_eeg_files()
        merge_ff_data()
        train_test_utils.train_ensemble(EEG_MODEL_TYPE, FF_MODEL_TYPE, ENSEMBLE_MODEL_TYPE, EEG_TASK,
                                        data_preprocessing=ENSEMBLE_PREPROC, folds=5,
                                        split_ratio=ENSEMBLE_SPLIT, verbosity=2)
        print("Done.")

    if args.train_ff:
        print("\n\n[ TRAIN FACIAL FEATURES MODEL ]\n\n")
        merge_ff_data()
        train_test_utils.train_ff(FF_MODEL_TYPE, data_preprocessing=FF_PREPROC, folds=3, split_ratio=FF_SPLIT,
                                  verbosity=2)
        print("Done.")
