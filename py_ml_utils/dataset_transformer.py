import pandas as pd
from py_ml_utils.feature_transformer import FeatureTransformation
from py_ml_utils.missing_value_inferer import MissingValueInferer
from collections import defaultdict


class FeatureTransformationPair(object):
    # 2 use-cases : train and test? Logic should be held in dataset_transformer ?
    def __init__(self,
                 transformer,
                 missing_inferer=None):
        # type: (FeatureTransformation, Any) -> None
        self.inferer = missing_inferer
        self.transformer = transformer

    def get_id(self):
        return self.transformer.feature_name + "|" + self.transformer.process_name


class DatasetTransformer(object):
    def __init__(self):
        pass

    def transform(self, train=None, test=None, target=None, folds=None, tf_pairs=None):

        # Create new datasets
        tf_train = pd.DataFrame()
        tf_test = pd.DataFrame()

        # Keep track of the indexing info
        trn_idx = train.index

        features_to_cols_dict = defaultdict(lambda: [])

        # Go through pairs
        for pair in tf_pairs:

            # Transform train
            if pair.inferer is not None:
                train[pair.inferer.name] = pair.inferer.infer(train)

            # Now call the transformation process
            if hasattr(pair.transformer, "get_oof_data"):
                ft_df = pair.transformer.get_oof_data(train, target, folds)
            else:
                ft_df = pair.transformer.fit_transform(train, target)

            if "DataFrame" in str(type(ft_df)):
                features_to_cols_dict[pair.get_id()] = [col for col in ft_df.columns]
            else:
                features_to_cols_dict[pair.get_id()] = [ft_df.name]

            # Include transformed data in dataset
            if len(tf_train) == 0:
                tf_train = pd.DataFrame(ft_df.copy())
            else:
                # Beware pd.concat will sort the index !
                tf_train = pd.concat([tf_train, ft_df], axis=1)
            del ft_df

            # Transform test
            if test is not None:
                # keep index since pd.concat will sort the index
                tst_idx = test.index

                if pair.inferer is not None:
                    train[pair.inferer.name] = pair.inferer.infer(train)
                    test[pair.inferer.name] = pair.inferer.infer(test)

                # Now call the transformation process
                ft_df = pair.transformer.fit(train, target)
                ft_df = pair.transformer.transform(test)

                if len(tf_test) == 0:
                    tf_test = pd.DataFrame(ft_df.copy())
                else:
                    tf_test = pd.concat([tf_test, ft_df], axis=1)
                del ft_df

        if test is not None:
            return tf_train.loc[trn_idx], tf_test.loc[tst_idx], features_to_cols_dict
        else:
            return tf_train.loc[trn_idx], None, features_to_cols_dict