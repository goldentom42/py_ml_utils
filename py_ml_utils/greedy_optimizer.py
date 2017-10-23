import numpy as np
import pandas as pd
# from py_ml_utils.feature_transformer import *
# from py_ml_utils.dataset_transformer import FeatureTransformationPair
import time


# TODO: would be nice to have a history of scores rounds as a column and features as a row
# Along with best score.

# TODO: set verbosity
# TODO add test suite for Greedy Optimizer


class GreedyOptimizer(object):

    def __init__(self):
        self.approved_features = None
        self.start = time.time()

    def optimize(self,
                 estimator,
                 criterion,
                 probabilities,
                 maximize,
                 folds,
                 feature_pairs,
                 dataset,
                 target,
                 max_rounds=20,
                 eps=1e-3):

        if self.approved_features:
            approved_features = self.approved_features
        else:
            approved_features = []

        # Make sure we don't test features in the approved features set
        # Remove features with same name from features to test before recursion
        features_to_test = feature_pairs.copy()
        approved_names = [feature.transformer.feature_name for feature in approved_features]
        features_to_test = [feature for feature in features_to_test
                            if feature.transformer.feature_name not in approved_names]

        return self.get_best_features(approved_features,
                                      features_to_test,
                                      estimator,
                                      criterion,
                                      probabilities,
                                      folds,
                                      maximize,
                                      dataset,
                                      target,
                                      max_rounds,
                                      eps,
                                      benchmark=None)

    def set_approved_transformations(self, feature_pairs):
        # Make sure feature_pairs is a list of TransformationPairs
        if "FeatureTransformationPair" in str(type(feature_pairs)):
            # Only one pair is provided, transform to list
            self.approved_features = [feature_pairs]
        elif "list" in str(type(feature_pairs)):
            # Check all items in the list are a FeatureTransformationPair
            for pair in feature_pairs:
                if "FeatureTransformationPair" not in str(type(pair)):
                    raise ValueError(
                        "Only a list of FeatureTransformationPair or a FeatureTransformationPair can be provided.")
            self.approved_features = feature_pairs
        else:
            raise ValueError("Only a list of FeatureTransformationPair or a FeatureTransformationPair can be provided.")

    @staticmethod
    def check_improvement(maximize, benchmark, score, eps):

        if benchmark is None:
            benchmark = score
            has_improved = True
        else:
            if maximize:
                if score - benchmark > eps:
                    has_improved = True
                    benchmark = score
                else:
                    has_improved = False

            else:
                if benchmark - score > eps:
                    has_improved = True
                    benchmark = score
                else:
                    has_improved = False

        return benchmark, has_improved

    def get_best_features(self,
                          approved_features,
                          features_to_test,
                          estimator,
                          criterion,
                          probabilities,
                          folds,
                          maximize,
                          dataset,
                          target,
                          max_rounds,
                          eps,
                          benchmark):

        if max_rounds == 0:
            return approved_features, benchmark

        # For each feature in features to test compute score
        scores = [self.get_score(approved_features + [feature],
                                 criterion,
                                 probabilities,
                                 dataset,
                                 target,
                                 folds,
                                 estimator)
                  for feature in features_to_test]

        # Get best score for this round
        if maximize:
            best_score_idx = np.argsort(scores)[::-1][0]
        else:
            best_score_idx = np.argsort(scores)[0]

        # Check improvement
        benchmark, has_improved = self.check_improvement(maximize, benchmark, scores[best_score_idx], eps)
        if not has_improved:
            return approved_features, benchmark

        # Update approved features and remove feature from features_to_test
        approved_features += [features_to_test[best_score_idx]]

        miss_name = "None"
        if features_to_test[best_score_idx].inferer:
            miss_name = features_to_test[best_score_idx].inferer.process_name
        print("benchmark : %.5f for %s | %s | %s"
              % (benchmark,
                 features_to_test[best_score_idx].transformer.feature_name,
                 features_to_test[best_score_idx].transformer.process_name,
                 miss_name))

        approved_names = [feature.transformer.feature_name for feature in approved_features]
        # print("Approved names : ", approved_names)

        # Remove features with same name from features to test before recursion
        features_to_test = [feature for feature in features_to_test
                            if feature.transformer.feature_name not in approved_names]

        # decrease max_rounds
        max_rounds -= 1

        # call get_best_score
        return self.get_best_features(approved_features,
                                      features_to_test,
                                      estimator,
                                      criterion,
                                      probabilities,
                                      folds,
                                      maximize,
                                      dataset,
                                      target,
                                      max_rounds,
                                      eps,
                                      benchmark)

    # @staticmethod
    def get_score(self, features,
                  criterion,
                  probabilities,
                  dataset,
                  target,
                  folds,
                  estimator):

        # Transform dataset
        trans_df = pd.DataFrame()
        for pair in features:

            # Check if there is a missing inference required
            # TODO I think this should be in FeaturePair
            if pair.inferer:
                dataset[pair.inferer.name] = pair.inferer.infer(dataset)

            # Now call the transformation process
            if hasattr(pair.transformer, "get_oof_data"):
                ft_df = pair.transformer.get_oof_data(dataset, target, folds)
            else:
                ft_df = pair.transformer.fit_transform(dataset, target)

            if len(trans_df) == 0:
                trans_df = pd.DataFrame(ft_df.copy())
            else:
                trans_df = pd.concat([trans_df, ft_df], axis=1)
            del ft_df

        # Init OOF data
        if probabilities:
            oof = np.zeros((len(dataset), len(target.unique())))
        else:
            oof = np.zeros(len(dataset))

        # Go through folds to compute score
        for trn_idx, val_idx in folds.split(trans_df, target):
            trn_x, trn_y = trans_df.iloc[trn_idx], target.iloc[trn_idx]
            val_x, val_y = trans_df.iloc[val_idx], target.iloc[val_idx]
            # Add support for LightGBM with early stopping
            if "LGBM" in str(estimator):
                estimator.fit(trn_x.values, trn_y.values,
                              eval_set=[(val_x.values, val_y.values)],
                              early_stopping_rounds=50,
                              # eval_metric="auc",
                              verbose=0)
                best_round = estimator.best_iteration_

                if probabilities:
                    z = estimator.predict_proba(val_x.values, num_iteration=best_round)
                    # print(z.shape)
                    if len(z.shape) == 1:
                        oof[val_idx, :] = np.hstack((
                            1 - estimator.predict_proba(val_x.values).reshape(-1, 1),
                            estimator.predict_proba(val_x.values).reshape(-1, 1)
                        ))
                    else:
                        oof[val_idx, :] = estimator.predict_proba(val_x.values)
                else:
                    oof[val_idx] = estimator.predict(val_x.values, num_iteration=best_round)
            elif "XGB" in str(estimator):
                estimator.fit(trn_x.values, trn_y.values,
                              eval_set=[(val_x.values, val_y.values)],
                              early_stopping_rounds=50,
                              verbose=False)
                best_round = estimator.best_ntree_limit

                if probabilities:
                    z = estimator.predict_proba(val_x.values, ntree_limit=best_round)
                    # print(z.shape)
                    if len(z.shape) == 1:
                        oof[val_idx, :] = np.hstack((
                            1 - estimator.predict_proba(val_x.values).reshape(-1, 1),
                            estimator.predict_proba(val_x.values).reshape(-1, 1)
                        ))
                    else:
                        oof[val_idx, :] = estimator.predict_proba(val_x.values)
                else:
                    oof[val_idx] = estimator.predict(val_x.values, ntree_limit=best_round)
            else:
                if trn_x.shape[1] <= 1:
                    estimator.fit(trn_x.values.reshape(-1, 1), trn_y.values)
                else:
                    estimator.fit(trn_x.values, trn_y.values)

                if probabilities:
                    z = estimator.predict_proba(val_x.values)
                    # print(z.shape)
                    if len(z.shape) == 1:
                        oof[val_idx, :] = np.hstack((
                            1 - estimator.predict_proba(val_x.values).reshape(-1, 1),
                            estimator.predict_proba(val_x.values).reshape(-1, 1)
                        ))
                    else:
                        oof[val_idx, :] = estimator.predict_proba(val_x.values)
                else:
                    oof[val_idx] = estimator.predict(val_x.values)

        # return score
        if probabilities and (len(oof.shape) > 1):
            # Check for binary classification
            if oof.shape[1] == 2:
                # Probability estimates can be reduced to the positive predictions
                # This insures the whole thing works with AUC, GINI...
                score = criterion(target, oof[:, 1])
        else:
            score = criterion(target, oof)

        print("%-40s : %.7f in %5.1f" % (features[-1].transformer.feature_name, score, ((time.time() - self.start) / 60)))
        return score