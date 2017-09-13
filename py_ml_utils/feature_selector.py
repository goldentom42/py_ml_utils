import pandas as pd
import numpy as np
from py_ml_utils.dataset_transformer import FeatureTransformationPair, DatasetTransformer
from collections import defaultdict
import time
from typing import Any, Callable


class FeatureSelector(object):

    def __init__(self, max_features=.5, max_runs=100):
        self.max_runs = max_runs
        self.max_features = max_features
        self.start_time = 0

    @staticmethod
    def _check_pairs(pairs):
        # Make sure feature_pairs is a list of TransformationPairs
        if "FeatureTransformationPair" in str(type(pairs)):
            # Only one pair is provided, transform to list
            return [pairs]
        elif "list" in str(type(pairs)):
            # Check all items in the list are a FeatureTransformationPair
            for pair in pairs:
                if "FeatureTransformationPair" not in str(type(pair)):
                    raise ValueError(
                        "Only a list of FeatureTransformationPair or a FeatureTransformationPair can be provided.")
            return pairs
        else:
            raise ValueError("Only a list of FeatureTransformationPair or a FeatureTransformationPair can be provided.")

    @staticmethod
    def _check_features_vs_dataframe(dataset, pairs):
        # type: (pd.DataFrame, [FeatureTransformationPair]) -> None

        # Check dataset type
        if "DataFrame" not in str(type(dataset)):
            raise ValueError("dataset must be provided as a pandas DataFrame")

        dataset_features = dataset.columns

        for pair in pairs:
            if pair.transformer.feature_name not in dataset_features:
                raise ValueError("Feature " + pair.transformer.feature_name + " is not in the dataset. "
                                 "Please check TransformationPairs and Dataset consistency")

        return None

    def _sample_features(self, pairs):
        # type: ([FeatureTransformationPair]) -> [FeatureTransformationPair]
        """ Randomly build a subsample of the transformations """

        # Create FeatureTransformationIndex
        idx = np.arange(len(pairs))
        # Shuffle the index
        np.random.shuffle(idx)
        # Return subsample
        if len(pairs) * self.max_features < 1:
            # Make sure there is at least 1 feature in the dataset
            nb_features = 1
        else:
            nb_features = round(len(pairs) * self.max_features)
        return [pairs[i] for i in idx[: nb_features]]

    @staticmethod
    def _create_dataset(dataset,
                        run_features,
                        target=None,
                        folds=None):
        # type: (pd.DataFrame, [FeatureTransformationPair], pd.Series, Any) -> pd.DataFrame
        """ Create dataset for the run using run_features transformations """

        dtf = DatasetTransformer()
        return dtf.transform(train=dataset,
                             test=None,
                             target=target,
                             folds=folds,
                             tf_pairs=run_features)

    @staticmethod
    def _get_score(estimator,
                   dataset,
                   target,
                   folds=None,
                   metric=None,
                   probability=False):
        # type: (Any, pd.DataFrame, pd.Series, Any, Callable, bool) -> (float, Any)

        # Init OOF data
        if probability:
            oof = np.zeros((len(dataset), len(target.unique())))
        else:
            oof = np.zeros(len(dataset))

        # Init Importances
        importances = np.zeros(dataset.shape[1])

        # Go through folds to compute score
        for trn_idx, val_idx in folds.split(dataset, target):
            # Split data into training and validation sets
            trn_x, trn_y = dataset.iloc[trn_idx], target.iloc[trn_idx]
            val_x, val_y = dataset.iloc[val_idx], target.iloc[val_idx]

            # Fit estimator
            if trn_x.shape[1] <= 1:
                estimator.fit(trn_x.values.reshape(-1, 1), trn_y.values)
            else:
                estimator.fit(trn_x.values, trn_y.values)

            # Update importances if available
            if hasattr(estimator, "feature_importances_"):
                importances += estimator.feature_importances_ / folds.n_splits

            # Get predictions
            if probability:
                oof[val_idx, :] = estimator.predict_proba(val_x)
            else:
                oof[val_idx] = estimator.predict(val_x)

        # return score
        if hasattr(estimator, "feature_importances_"):
            return metric(target, oof), importances
        else:
            return metric(target, oof), None

    @staticmethod
    def _update_scores(features_score, pairs, score, imp, run_cols, feat_to_cols):
        # type: (defaultdict, [FeatureTransformationPair], float, Any, Any, dict) -> defaultdict
        """
        :param features_score: object containing all features' scores and importances (if available)
        :param pairs: set of features used during the current run
        :param score: score obatined by the classifier during the current run
        :param imp: feature importances provided by the estimator for the current run,
                    None if estimator does not have feature importances
        :param run_cols: columns used during current run
        :param feat_to_cols: dictionary linking each feature transformation to dataframe columns
                             this is useful for dummy transformation where 1 transformation is linked to several
                             dummy columns. It serves aggregating feature_importance back to feature transformations
        :return: updated feature scores
        """
        # Get importances for each columns
        importances = defaultdict()
        if imp is not None:
            for i, col in enumerate(run_cols):
                importances[col] = imp[i]

        for i, feature in enumerate(pairs):
            # Create features_id
            feature_id = FeatureSelector._get_feature_id(feature)
            features_score[feature_id]["name"] = feature.transformer.feature_name
            features_score[feature_id]["process"] = feature.transformer.process_name
            features_score[feature_id]["shadow"] = feature.transformer.shadow
            features_score[feature_id]["score"] += score
            features_score[feature_id]["count"] += 1

            if imp is not None:
                # Retrieve list of columns for current feature
                list_of_cols = feat_to_cols[feature.get_id()]
                # Sum up importances of all columns for current feature transformation
                for col in list_of_cols:
                    features_score[feature_id]["importance"] += importances[col]

        features_score["_Mean_score_"]["name"] = "Mean score"
        features_score["_Mean_score_"]["score"] += score
        features_score["_Mean_score_"]["count"] += 1
        if imp is not None:
            features_score["_Mean_score_"]["importance"] += np.mean(imp)

        return features_score

    @staticmethod
    def _get_feature_id(feature):
        if feature.transformer.shadow:
            feature_id = "Shadow"
        else:
            feature_id = ""
        feature_id += "_N:" + feature.transformer.feature_name
        feature_id += "_P:" + feature.transformer.process_name
        if feature.inferer:
            feature_id += "_MISS:" + feature.inferer.missing_process
        return feature_id

    @staticmethod
    def _build_features_recap(feature_scores, maximize):
        names = [feature_scores[key]["name"] for key in feature_scores.keys()]
        processes = [feature_scores[key]["process"] for key in feature_scores.keys()]
        shadows = [feature_scores[key]["shadow"] for key in feature_scores.keys()]
        scores = [feature_scores[key]["score"] / feature_scores[key]["count"] for key in feature_scores.keys()]
        importances = [feature_scores[key]["importance"] / feature_scores[key]["count"]
                       for key in feature_scores.keys()]
        counts = [feature_scores[key]["count"] for key in feature_scores.keys()]
        full_data = pd.DataFrame()
        full_data["feature"] = names
        full_data["process"] = processes
        full_data["shadow"] = shadows
        full_data["score"] = scores
        full_data["importance"] = importances / (np.sum(importances) + 1e-7)
        full_data["occurences"] = counts

        return full_data.sort_values(by="score", ascending=(not maximize))

    def select(self,
               dataset=None,
               target=None,
               pairs=[],
               estimator=None,
               metric=None,
               probability=False,
               folds=None,
               maximize=True):
        # type: (pd.DataFrame, pd.Series, [FeatureTransformationPair], Any, Any, bool, Any, bool) -> int

        # Get start time
        self.start_time = time.time()

        # Check pairs
        pairs = self._check_pairs(pairs)

        # Check paris against DataFrame
        self._check_features_vs_dataframe(dataset, pairs)

        # Features score is probably made of a cumulated score, number of runs and mean score for each feature
        feature_scores = defaultdict(lambda: {"name": "",
                                              "process": "",
                                              "shadow": True,
                                              "count": 0,
                                              "score": 0.0,
                                              "importance": 0.0})

        for run in range(self.max_runs):
            # print("coucou")
            print("Run #%-5d @ %5.1f min" % (run + 1, (time.time() - self.start_time) / 60), end='')

            # Sample Features
            run_features = self._sample_features(pairs)

            # Create Dataset
            run_dataset, _, feat_to_cols = self._create_dataset(dataset, run_features, target, folds)
            run_cols = run_dataset.columns

            # Compute score
            run_score, run_imp = self._get_score(estimator, run_dataset, target, folds, metric, probability)

            # Update Feature scores
            feature_scores = self._update_scores(feature_scores,
                                                 run_features,
                                                 run_score,
                                                 run_imp,
                                                 run_cols,
                                                 feat_to_cols)
            print('\r' * 22, end='')

        return self._build_features_recap(feature_scores, maximize)
