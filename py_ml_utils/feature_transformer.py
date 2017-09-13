"""
@author Olivier Grellier
@email goldentom42@gmail.com
"""
import pandas as pd
import numpy as np
import warnings
import inspect
from py_ml_utils.missing_value_inferer import MissingValueInferer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LogisticRegression


class FeatureTransformation(object):
    """
    A new transformation need to
    - Implement __init__ if needed and call super init
    - Implement _fit_special_process
    - Implement _transform_special_process
    """

    def __init__(self, feature_name=None, shadow=False):
        self._name = feature_name
        self.fitted = False
        self._process_name = None
        self._shadow = shadow

    @property
    def feature_name(self):
        return self._name

    @property
    def process_name(self):
        return self._process_name

    @property
    def shadow(self):
        return self._shadow

    def _fit_special_process(self, data, target=None):
        raise NotImplementedError()

    def fit(self, data, target=None):
        # Check feature is in data
        if self._name not in data.columns:
            raise ValueError("Feature " + self._name + " is not in the dataset")
        # Call Transformation specific process
        self._fit_special_process(data=data, target=target)
        # Set fitted
        self.fitted = True

    def fit_transform(self, data, target=None):
        # Call fit
        self.fit(data=data, target=target)
        # Call transform
        return self.transform(data)

    def _transform_special_process(self, data):
        raise NotImplementedError()

    def transform(self, data):
        if self.fitted is not True:
            raise NotFittedError("Transformation is not fitted yet.")
        # Check if shadow shuffling process has to be used
        if self._shadow:
            temp = self._transform_special_process(data)
            if "dataframe" in str(type(temp)).lower():
                z = np.array(temp)
                idx = np.arange(len(z))
                np.random.shuffle(idx)
                return pd.DataFrame(z[idx],
                                    columns=temp.columns,
                                    index=temp.index)
            else:
                z = np.array(temp)
                np.random.shuffle(z)
                return pd.Series(z, name="shadow_" + self._name, index=temp.index)
        else:
            return self._transform_special_process(data)


class IdentityTransformation(FeatureTransformation):
    def __init__(self, feature_name=None, shadow=False):
        # Call super
        super(IdentityTransformation, self).__init__(feature_name=feature_name, shadow=shadow)
        self._process_name = "Identity"

    def _fit_special_process(self, data, target=None):
        pass

    def _transform_special_process(self, data):
        return data[self._name]


class ShadowTransformation(FeatureTransformation):
    def __init__(self, feature_name=None):
        # Call super
        super(ShadowTransformation, self).__init__(feature_name)
        self._process_name = "Shadow"

    def _fit_special_process(self, data, target=None):
        pass

    def _transform_special_process(self, data):
        z = data[[self._name]].copy()
        vals = np.array(data[self._name])
        np.random.shuffle(vals)
        z[self._name] = vals
        return pd.Series(z[self._name], name="shadow_" + self._name, index=z.index)


class ExpoTransformation(FeatureTransformation):
    def __init__(self, feature_name=None):
        # Call super
        super(ExpoTransformation, self).__init__(feature_name)
        self._process_name = "Exponential"

    def _fit_special_process(self, data, target=None):
        pass

    def _transform_special_process(self, data):
        return np.exp(data[self._name])


class PowerTransformation(FeatureTransformation):
    def __init__(self, feature_name=None, power=2):
        # Call super
        super(PowerTransformation, self).__init__(feature_name)
        self.power = power
        self._process_name = "Power_" + str(power)

    def _fit_special_process(self, data, target=None):
        pass

    def _transform_special_process(self, data):
        return np.power(data[self._name], self.power)


class InversePowerTransformation(FeatureTransformation):
    def __init__(self, feature_name=None, power=1, epsilon=1e-5):
        # Call super
        super(InversePowerTransformation, self).__init__(feature_name)
        self.power = power
        self.epsilon = epsilon
        self._process_name = "Inverse_Power_" + str(power)

    def _fit_special_process(self, data, target=None):
        # Check for zeros
        if (data[self._name] == 0).sum() > 0:
            warnings.warn(self._name + " series contain 0s. 1e-5 has been added before inversion.", category=UserWarning)

    def _transform_special_process(self, data):
        return 1 / np.power(data[self._name].replace(0, self.epsilon), self.power)


class RootTransformation(FeatureTransformation):
    def __init__(self, feature_name=None, power=2):
        # Call super
        super(RootTransformation, self).__init__(feature_name)
        self.power = power
        self._process_name = "Root_power_" + str(power)

    def _fit_special_process(self, data, target=None):
        # Check for negative data
        if (data[self._name] < 0).sum() > 0:
            raise ValueError(self._name + " series contain negative values.")

    def _transform_special_process(self, data):
        return np.power(data[self._name], self.power)


class LogTransformation(FeatureTransformation):
    def __init__(self, feature_name=None, epsilon=1e-5):
        # Call super
        super(LogTransformation, self).__init__(feature_name)
        self.epsilon = epsilon
        self._process_name = "Log"

    def _fit_special_process(self, data, target=None):
        # Check for negative data
        if (data[self._name] < 0).sum() > 0:
            raise ValueError(self._name + " series contain negative values.")
        # Check for zeros
        if (data[self._name] == 0).sum() > 0:
            warnings.warn(self._name + " series contain 0s. 1e-5 has been added before log processing.",
                          category=UserWarning)

    def _transform_special_process(self, data):
        return np.log(data[self._name].replace(0, self.epsilon))


class OOFTransformation(FeatureTransformation):  # Or Object
    """
    OOF transformations use k-folding
    The problem is that usually training sets use k-folding and test sets don't
    Folds should be kept outside of the transformation
    """
    def get_oof_data(self, data, target, folds):
        # raise NotImplementedError()
        oof = np.zeros(len(data))
        ft_name = ""
        for trn_idx, val_idx in folds.split(data, target):
            trn_X, trn_Y = data[[self._name]].iloc[trn_idx], target.iloc[trn_idx]
            val_X, val_Y = data[[self._name]].iloc[val_idx], target.iloc[val_idx]
            self.fit(data=trn_X, target=trn_Y)
            ft_series = self.transform(data=val_X)
            ft_name = ft_series.name
            oof[val_idx] = ft_series
        # Return with additional noise
        return pd.Series(oof, name=ft_series.name, index=data.index)


class TargetAverageTransformation(OOFTransformation):

    MEAN = 'MEAN'
    MEDIAN = 'MEDIAN'

    def __init__(self, feature_name=None, average=MEAN, noise_level=0):
        # Call super
        super(TargetAverageTransformation, self).__init__(feature_name)
        self._process_name = "Target_Average_" + average
        # Keep average type
        self.average = average
        self.noise_level = noise_level
        # Place-holder for averages
        self.averages = None
        self.full_average = None
        self.target_name = None

    def _fit_special_process(self, data, target=None):
        # Compute averages
        if self.average == self.MEAN:
            self.averages = pd.concat([data[self._name], target], axis=1).groupby(by=self._name).mean()[target.name]
            self.full_average = target.mean()
        elif self.average == self.MEDIAN:
            self.averages = pd.concat([data[self._name], target], axis=1).groupby(by=self._name).median()[target.name]
            self.full_average = target.median()
        else:
            raise ValueError("average_type must be one of 'MEAN' of 'MEDIAN'")

        self.target_name = target.name

    def _transform_special_process(self, data):
        ft_series = pd.merge(
            pd.DataFrame(data[self._name]),
            self.averages.reset_index().rename(columns={'index': self.target_name, self.target_name: 'average'}),
            on=self._name,
            how='left')['average'].rename(self._name + '_' + self.average.lower()).fillna(self.full_average)

        return ft_series * (1 + self.noise_level * np.random.randn(len(ft_series)))


class FrequencyTransformation(OOFTransformation):

    def __init__(self, feature_name=None, noise_level=None, shadow=False):
        # Call super
        super(FrequencyTransformation, self).__init__(feature_name, shadow)
        self._process_name = "Frequency"
        # Keep noise_level
        self.noise_level = noise_level
        # Place-holder for frequencies
        self.occurencies = None

    def _fit_special_process(self, data, target=None):
        # Compute occurences
        self.occurences = data[self._name].value_counts(dropna=False) / len(data)
        self.occurences = self.occurences.reset_index().rename(columns={'index': self._name,
                                                                        self._name: 'occurence'})

    def _transform_special_process(self, data):
        # Beware : pd.merge reset the index
        temp = pd.merge(pd.DataFrame(data[self._name]),
                        self.occurences,
                        on=self._name,
                        how='left')['occurence'].rename(self._name + '_freq')
        nan_mean = temp.isnull().mean()
        temp.fillna(nan_mean, inplace=True)
        temp.index = data.index
        if self.noise_level:
            temp *= (1 + self.noise_level * np.random.randn(len(temp)))

        return temp


class DummyTransformation(OOFTransformation):
    def __init__(self, feature_name=None, drop_level=None, noise_level=None, keep_dum_cols_with_nan=True, shadow=False):
        # Call super
        super(DummyTransformation, self).__init__(feature_name, shadow)
        self._process_name = "Dummy"
        # Keep noise_level
        self.noise_level = noise_level
        self.drop_level = drop_level
        self.keep_dum_cols_with_nan = keep_dum_cols_with_nan

    def _fit_special_process(self, data, target=None):
        pass

    def _transform_special_process(self, data):
        series = data[self._name]

        # First get the features count
        counts = series.value_counts(dropna=False) / len(series)
        counts = counts.reset_index().rename(columns={'index': series.name, series.name: 'freq'})

        # Get values we need to keep
        counts['new_name'] = counts[series.name].astype(str)
        if self.drop_level:
            counts.loc[counts['freq'] < self.drop_level, 'new_name'] = 'other'

        # Transform to dict
        counts[series.name] = counts[series.name].astype(str)
        conversion = counts[[series.name, 'new_name']].set_index(series.name).to_dict()['new_name']

        # Convert series to dummy
        return pd.get_dummies(series.apply(lambda x: conversion[str(x)]), prefix=series.name)

    def get_oof_data(self, data, target, folds):
        """
        Dummy transformation requires a special oof process where columns are checked before returning the matrix
        Note that the index may be shuffled
        :param data:
        :param target:
        :param folds:
        :return:
        """
        # raise NotImplementedError()
        oof = pd.DataFrame()
        idx = data.index
        for trn_idx, val_idx in folds.split(data, target):
            # Create data partitions
            trn_X, trn_Y = data[[self._name]].iloc[trn_idx], target.iloc[trn_idx]
            val_X, val_Y = data[[self._name]].iloc[val_idx], target.iloc[val_idx]
            # Fit has no effect
            self.fit(data=trn_X, target=trn_Y)
            val_dummies = self.transform(data=val_X)
            val_dummies.index = val_X.index
            oof = pd.concat([oof, val_dummies], axis=0)

        # At this point NaN could be replaced by zeros and this is the standard dummy transform
        # Or we could drop columnes that contains NaN
        if self.keep_dum_cols_with_nan:
            return oof.fillna(0)
        else:
            # Find cols with NaN
            cols_with_NaN = oof.isnull().sum(axis=0) > 0
            cols_with_NaN = list(cols_with_NaN[cols_with_NaN].index.values)
            # Drop these columns
            return oof.drop(cols_with_NaN, axis=1)


class RegressorTransformation(OOFTransformation):
    # we may need to make a transformation first : label encoding, frequency, dummy or average
    def __init__(self,
                 feature_name=None,
                 regressor=Lasso(),
                 noise_level=None):

        # Call super
        super(RegressorTransformation, self).__init__(feature_name)
        self._process_name = "Regressor"
        # Keep average type
        self.noise_level = noise_level
        # Place-holder for averages
        self.regressor = regressor

    def _fit_special_process(self, data, target=None):
        self.regressor.fit(data[[self._name]],
                           target)

    def _transform_special_process(self, data):
        ft_series = pd.Series(self.regressor.predict(data[[self._name]]),
                              name=self._name,
                              index=data.index)
        if self.noise_level:
            ft_series *= (1 + self.noise_level * np.random.randn(len(ft_series)))

        return ft_series


class CategoricalRegressorTransformation(OOFTransformation):
    # This transformation does not fit the others for the following reasons :
    #  1. A dummy transformation may return different number of columns for fit and transform if data is different
    #  2. We need to fix this since predict and fit cannot run on different columns !
    #  3. fit needs to know on what data you transform on and transform need to know on which data you fit on
    #  4. OOF would work since you would call dummy transformation on the whole data when det_oof_data is called
    #  5. The real problem is for fit and transform in an independent process...
    #  6. The last use-case would work for sufficiently high drop_level threshold
    def __init__(self,
                 feature_name=None,
                 regressor=Lasso(),
                 noise_level=None,
                 drop_level=None,
                 keep_dum_cols_with_nan=True):

        # Call super
        super(CategoricalRegressorTransformation, self).__init__(feature_name)
        self._process_name = "Categorical_Regressor"
        # Keep average type
        self.noise_level = noise_level
        # Place-holder for averages
        self.regressor = regressor
        self.dum_tf = DummyTransformation(feature_name=feature_name,
                                          drop_level=drop_level,
                                          noise_level=None,
                                          keep_dum_cols_with_nan=keep_dum_cols_with_nan)
        self.oof_process = False
        self.fit_columns = None

    def fit(self, data, target=None):
        # Call Transformation specific process
        self._fit_special_process(data=data, target=target)
        # Set fitted
        self.fitted = True

    def _fit_special_process(self, data, target=None):
        if self.oof_process:
            # we are called as part of an OOF process
            self.regressor.fit(data, target)
        else:
            # We are called independently
            dum_data = self.dum_tf.fit_transform(data=data)
            self.fit_columns = dum_data.columns
            self.regressor.fit(dum_data, target)

    def _transform_special_process(self, data):
        if self.oof_process:
            ft_series = pd.Series(self.regressor.predict(data),
                                  name=self._name,
                                  index=data.index)
        else:
            dum_data = self.dum_tf.fit_transform(data=data)
            # Check if we have same columns as in fit
            if set(self.fit_columns) != set(dum_data.columns):
                raise ValueError("Dummy columns discrepency. Increase drop_level.")

            ft_series = pd.Series(self.regressor.predict(dum_data),
                                  name=self._name,
                                  index=data.index)
        if self.noise_level:
            ft_series *= (1 + self.noise_level * np.random.randn(len(ft_series)))

        return ft_series

    def get_oof_data(self, data, target, folds):
        # First call
        dum_data = self.dum_tf.get_oof_data(data=data, target=target, folds=folds)
        self.oof_process = True
        # raise NotImplementedError()
        oof = np.zeros(len(data))
        for trn_idx, val_idx in folds.split(dum_data, target):
            trn_X, trn_Y = dum_data.iloc[trn_idx], target.iloc[trn_idx]
            val_X, val_Y = dum_data.iloc[val_idx], target.iloc[val_idx]
            self.fit(data=trn_X, target=trn_Y)
            oof[val_idx] = self.transform(data=val_X)

        self.oof_process = False

        # Return with additional noise
        return pd.Series(oof, name=self._name, index=data.index)


class ClassifierTransformation(OOFTransformation):
    def __init__(self, feature_name=None, classifier=LogisticRegression(), probabilities=True, noise_level=0):
        # Call super
        super(ClassifierTransformation, self).__init__(feature_name)
        self._process_name = "Classifier"
        # Keep average type
        self.noise_level = noise_level
        # Place-holder for averages
        self.classifier = classifier
        self.probabilities = probabilities

    def _fit_special_process(self, data, target=None):
        self.classifier.fit(data[[self._name]], target)

    def _transform_special_process(self, data):
        if self.probabilities:
            return pd.Series(self.classifier.predict(data[[self._name]]))
        else:
            return pd.DataFrame(self.classifier.predict_proba(data[[self._name]]))


class CategoricalClassifierTransformation(OOFTransformation):
    # This transformation does not fit the others for the following reasons :
    #  1. A dummy transformation may return different number of columns for fit and transform if data is different
    #  2. We need to fix this since predict and fit cannot run on different columns !
    #  3. fit needs to know on what data you transform on and transform need to know on which data you fit on
    #  4. OOF would work since you would call dummy transformation on the whole data when det_oof_data is called
    #  5. The real problem is for fit and transform in an independent process...
    #  6. The last use-case would work for sufficiently high drop_level threshold
    def __init__(self,
                 feature_name=None,
                 classifier=LogisticRegression(),
                 probabilities=True,
                 noise_level=None,
                 drop_level=None,
                 keep_dum_cols_with_nan=True):

        # Call super
        super(CategoricalClassifierTransformation, self).__init__(feature_name)
        self._process_name = "Categorical_Classifier"
        # Keep average type
        self.noise_level = noise_level
        # Place-holder for averages
        self.classifier = classifier
        self.probabilities = probabilities
        self.dum_tf = DummyTransformation(feature_name=feature_name,
                                          drop_level=drop_level,
                                          noise_level=None,
                                          keep_dum_cols_with_nan=keep_dum_cols_with_nan)
        self.oof_process = False
        self.fit_columns = None

    def fit(self, data, target=None):
        # Call Transformation specific process
        self._fit_special_process(data=data, target=target)
        # Set fitted
        self.fitted = True

    def _fit_special_process(self, data, target=None):
        if self.oof_process:
            # we are called as part of an OOF process
            self.classifier.fit(data, target)
        else:
            # We are called independently
            dum_data = self.dum_tf.fit_transform(data=data)
            self.fit_columns = dum_data.columns
            self.classifier.fit(dum_data, target)

    def get_proba_columns(self):
        return ["proba_" + str(c) for c in self.classifier.classes_]

    def _transform_special_process(self, data):
        if self.oof_process:
            if self.probabilities:
                ft_series = pd.DataFrame(self.classifier.predict_proba(data),
                                         columns=self.get_proba_columns(),
                                         index=data.index)
            else:
                ft_series = pd.Series(self.classifier.predict(data),
                                      name=self._name,
                                      index=data.index)
        else:
            dum_data = self.dum_tf.fit_transform(data=data)
            # Check if we have same columns as in fit
            if set(self.fit_columns) != set(dum_data.columns):
                raise ValueError("Dummy columns discrepency. Increase drop_level.")

            if self.probabilities:
                ft_series = pd.DataFrame(self.classifier.predict_proba(dum_data),
                                         columns=self.get_proba_columns(),
                                         index=data.index)
            else:
                ft_series = pd.Series(self.classifier.predict(dum_data),
                                      name=self._name,
                                      index=data.index)
        if self.noise_level:
            ft_series *= (1 + self.noise_level * np.random.randn(len(ft_series)))

        return ft_series

    def get_oof_data(self, data, target, folds):
        # First call
        dum_data = self.dum_tf.get_oof_data(data=data, target=target, folds=folds)
        self.oof_process = True
        # raise NotImplementedError()
        if self.probabilities:
            oof = np.zeros((len(data), len(np.unique(target))))
        else:
            oof = np.zeros(len(data))
        for trn_idx, val_idx in folds.split(dum_data, target):
            trn_X, trn_Y = dum_data.iloc[trn_idx], target.iloc[trn_idx]
            val_X, val_Y = dum_data.iloc[val_idx], target.iloc[val_idx]
            self.fit(data=trn_X, target=trn_Y)
            if self.probabilities:
                oof[val_idx, :] = self.transform(data=val_X)
            else:
                oof[val_idx] = self.transform(data=val_X)

        self.oof_process = False

        # Return with additional noise
        if self.probabilities:
            return pd.DataFrame(oof,
                                columns=self.get_proba_columns(),
                                index=data.index)
        else:
            return pd.Series(oof, name=self._name, index=data.index)


class LabelEncodingTransformation(OOFTransformation):
    """
    Labels are sorted prior to encoding so that OOF has a chance to work
    """
    def __init__(self, feature_name=None, noise_level=0, shadow=False):
        # Call super
        super(LabelEncodingTransformation, self).__init__(feature_name=feature_name, shadow=shadow)
        self._process_name = "Label_Encoding"
        # Keep average type
        self.noise_level = noise_level
        self.encoder = None

    def _fit_special_process(self, data, target=None):
        _, self.encoder = pd.factorize(data[self._name], sort=True)

    def _transform_special_process(self, data):
        ft_series = pd.Series(self.encoder.get_indexer(data[self._name]),
                              name=self._name,
                              index=data.index)
        if self.noise_level:
            ft_series *= (1 + self.noise_level * np.random.randn(len(ft_series)))

        return ft_series

