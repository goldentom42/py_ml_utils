from unittest import TestCase
from py_ml_utils.feature_transformer import *
from sklearn.model_selection import KFold
import os.path


def get_path(file_name):
    """ Ensure file path is correct wherever the tests are called from """
    my_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(my_path, file_name)


class TestFeatureTransformer(TestCase):
    """
    Set of tests for FeatureTransformation objects
    """
    def test__feature_transform_identity(self):
        """ Test IdentityTransformation """
        series = pd.Series(np.random.choice(3, 20, p=[0.40, 0.40, 0.20]))
        ft = IdentityTransformation(feature_name="test")
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        # ft_series = ft._feature_transform_same(series=series)
        self.assertAlmostEqual(0, np.sum(np.abs(series - ft_series)))

    def test__feature_transform_square(self):
        """ Test PowerTransformation with power 2 """
        series = pd.Series(np.random.choice(3, 20, p=[0.40, 0.40, 0.20]))
        ft = PowerTransformation(feature_name="test", power=2)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        residual = (np.power(series,  2) - ft_series).abs().sum()
        self.assertAlmostEqual(0, residual)

    def test__feature_transform_power3(self):
        """ Test PowerTransformation with power 3 """
        series = pd.Series(np.random.choice(3, 20, p=[0.40, 0.40, 0.20]))
        ft = PowerTransformation(feature_name="test", power=3)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        residual = (np.power(series,  3) - ft_series).abs().sum()
        self.assertAlmostEqual(0, residual)

    def test__feature_transform_inverse(self):
        """ Test InverseTransformation with non zero values """
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = InversePowerTransformation(feature_name="test", power=1)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        self.assertAlmostEqual(0, np.sum(np.abs(1 / series - ft_series)))

    def test__feature_transform_inverse_with_zero(self):
        """ Test InverseTransformation with zeros, zeros are changed to 1e-5 """
        np.random.seed(13)
        series = pd.Series(np.random.choice([0, 1, 2], 20, p=[0.40, 0.40, 0.20]))
        ft = InversePowerTransformation(feature_name="test", power=1)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        np.random.seed(13)
        series = pd.Series(np.random.choice([1e-5, 1, 2], 20, p=[0.40, 0.40, 0.20]))
        self.assertAlmostEqual(0, np.sum(np.abs(1 / series - ft_series)))

    # def test__feature_transform_inverse_with_zero_check_warning(self):
    #     """ Test InverseTransformation with zeros and ensures UserWarning is issued"""
    #     np.random.seed(13)
    #     series = pd.Series(np.random.choice([0, 1, 2], 20, p=[0.40, 0.40, 0.20]))
    #     ft = FeatureTransformer()
    #     self.assertWarns(UserWarning, ft._feature_transform_inverse, series=series)

    def test__feature_transform_inverse_power2(self):
        """ Test InverseTransformation with power 2 """
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = InversePowerTransformation(feature_name="test", power=2)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        self.assertAlmostEqual(0, np.sum(np.abs(1 / (series ** 2) - ft_series)))

    def test__feature_transform_inverse_power3(self):
        """ Test InverseTransformation with power 3 """
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = InversePowerTransformation(feature_name="test", power=3)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        self.assertAlmostEqual(0, np.sum(np.abs(1 / (series ** 3) - ft_series)))

    def test__feature_transform_square_root(self):
        """ Test RootTransformation with power .5 """
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = RootTransformation(feature_name="test", power=.5)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        self.assertAlmostEqual(0, np.sum(np.abs(series ** .5) - ft_series))

    def test__feature_transform_square_root_negative_samples(self):
        """ Test RootTransformation with negative samples, Exception should be thrown """
        series = pd.Series(np.random.choice([-1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = RootTransformation(feature_name="test", power=.5)
        self.assertRaises(ValueError, ft.fit_transform, data=pd.DataFrame(series, columns=["test"]))

    def test__feature_transform_power_third(self):
        """ Test RootTransformation with power 1/3 """
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = RootTransformation(feature_name="test", power=1/3)
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        self.assertAlmostEqual(0, np.sum(np.abs(series ** (1/3) - ft_series)))

    def test__feature_transform_expo(self):
        """ Test ExpoTransformation """
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = ExpoTransformation(feature_name="test")
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        self.assertAlmostEqual(0, np.sum(np.abs(np.exp(series) - ft_series)))

    def test__feature_transform_log(self):
        """ Test LogTransformation """
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = LogTransformation(feature_name="test")
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        self.assertAlmostEqual(0, np.sum(np.abs(np.log(series) - ft_series)))

    def test__feature_transform_log_index(self):
        """ Test LogTransformation  and make sure index for output Series is equal to index of input Series"""
        series = pd.Series(np.random.choice([1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        idx = np.arange(20)
        np.random.shuffle(idx)
        series.index = idx
        ft = LogTransformation(feature_name="test")
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        # print(ft_series)
        self.assertEqual(0, np.sum(np.abs(series.index - ft_series.index)))

    def test__feature_transform_log_with_zeros(self):
        """ Test LogTransformation with zeros (changed to 1e-5) """
        np.random.seed(17)
        series = pd.Series(np.random.choice([0, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = LogTransformation(feature_name="test")
        ft_series = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        np.random.seed(17)
        series = pd.Series(np.random.choice([1e-5, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        self.assertAlmostEqual(0, np.sum(np.abs(np.log(series) - ft_series)))

    def test__feature_transform_log_with_negative_samples(self):
        """ Test LogTransformation with negative samples, exception is thrown """
        series = pd.Series(np.random.choice([-1, 2, 4], 20, p=[0.40, 0.40, 0.20]))
        ft = LogTransformation(feature_name="test")
        self.assertRaises(ValueError, ft.fit_transform, data=pd.DataFrame(series, columns=["test"]))

    def test__feature_transform_mean_average(self):
        """ Test TargetAverageTransformation for Out Of Fold and mean average, tests output values and index """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice([1, 2, 4], len_series, p=[0.40, 0.40, 0.20]), name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = TargetAverageTransformation(feature_name="test",
                                         average=TargetAverageTransformation.MEAN,
                                         noise_level=0)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        # check resulting series and index
        expected_results = pd.Series([0.25, 0.25, 0.0, 0.25, 0.0,
                                      0.25, 0.25, 0.0, 0.25, 0.25,
                                      0.25, 1./3., 0.25, 2./3., 2./3.,
                                      2./3., 0.25, 1./3., 2./3., 0.25], index=idx)
        self.assertAlmostEqual(0, (ft_series - expected_results).abs().sum(), places=5)

    def test__feature_transform_median_average(self):
        """ Test TargetAverageTransformation for Out Of Fold and median average, tests output values and index """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice([1, 2, 4], len_series, p=[0.40, 0.40, 0.20]), name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = TargetAverageTransformation(feature_name="test",
                                         average=TargetAverageTransformation.MEDIAN,
                                         noise_level=0)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        # check resulting series and index
        expected_results = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 1.0, 1.0,
                                      1.0, 0.0, 0.0, 1.0, 0.0], index=idx)
        self.assertAlmostEqual(0, (ft_series - expected_results).abs().sum(), places=5)

    def test__feature_transform_to_frequency(self):
        """ Test FrequencyTransformation OOF data computation """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice([1, 2, 4], len_series, p=[0.40, 0.40, 0.20]), name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = FrequencyTransformation(feature_name="test", noise_level=0)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        # check resulting series and index
        expected_results = pd.Series([0.4, 0.4, 0.2, 0.4, 0.2,
                                      0.4, 0.4, 0.2, 0.4, 0.4,
                                      0.4, 0.3, 0.4, 0.3, 0.3,
                                      0.3, 0.4, 0.3, 0.3, 0.4], index=idx)
        self.assertAlmostEqual(0, (ft_series - expected_results).abs().sum(), places=5)

    def test__feature_transform_to_frequency_with_nan(self):
        """ Test FrequencyTransformation OOF data computation, ensures frequency is also computed for NaNs
            values and index are tested """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice([1, 2, np.nan], len_series, p=[0.40, 0.40, 0.20]), name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = FrequencyTransformation(feature_name="test", noise_level=0)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)

        # check resulting series and index
        expected_results = pd.Series([0.4, 0.4, 0.2, 0.4, 0.2,
                                      0.4, 0.4, 0.2, 0.4, 0.4,
                                      0.4, 0.3, 0.4, 0.3, 0.3,
                                      0.3, 0.4, 0.3, 0.3, 0.4], index=idx)
        self.assertAlmostEqual(0, (ft_series - expected_results).abs().sum(), places=5)
        self.assertEqual("test_freq", ft_series.name)

    def test__feature_transform_to_frequency_fit_transform(self):
        """ Test FrequencyTransformation on the fit and transform process, values and index are tested """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice([1, 2, 5], len_series, p=[0.40, 0.40, 0.20]), name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Call feature transformer
        ft = FrequencyTransformation(feature_name="test", noise_level=0)
        ft.fit(data=series.loc[idx[:10]].to_frame(name="test"),
               target=target.loc[idx[:10]])
        ft_series = ft.transform(series.loc[idx[10:]].to_frame(name="test"))
        # check resulting series and index
        expected_results = pd.Series([0.4, 0.3, 0.4, 0.3, 0.3,
                                      0.3, 0.4, 0.3, 0.3, 0.4], index=idx[10:])
        self.assertAlmostEqual(0, (ft_series - expected_results).abs().sum(), places=5)
        self.assertEqual("test_freq", ft_series.name)

    def test__feature_transform_to_frequency_with_noise(self):
        """ Test FrequencyTransformation OOF process with additional noise, values and index are tested """
        # Create series and target
        len_series = 100
        np.random.seed(18)
        series = pd.Series(np.random.choice([1, 2, 4], len_series, p=[0.40, 0.40, 0.20]), name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = FrequencyTransformation(feature_name="test", noise_level=0)
        noise_less_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        ft = FrequencyTransformation(feature_name="test", noise_level=0.1)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        self.assertAlmostEqual(0.1, np.std(ft_series/noise_less_series), places=1)

    def test__feature_transform_oof_label_encoding(self):
        """ Test LabelEncodingTransformation OOF process, values and index are tested """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = LabelEncodingTransformation(feature_name="test", noise_level=0)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        # check resulting series and index
        expected_results = pd.Series([1, 1, 2, 0, 2,
                                      1, 1, -1, 0, 0,
                                      1, 2, 1, 0, 0,
                                      0, 1, 2, 0, 1], index=idx)
        self.assertAlmostEqual(0, (ft_series - expected_results).abs().sum(), places=5)

    def test__feature_transform_label_encoding_fit_transform(self):
        """ Test LabelEncodingTransformation fit/transform process, values and index are tested """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Call feature transformer
        ft = LabelEncodingTransformation(feature_name="test", noise_level=0)
        ft.fit(data=series.loc[idx[:10]].to_frame(name="test"),
               target=target.loc[idx[:10]])
        ft_series = ft.transform(series.loc[idx[10:]].to_frame(name="test"))
        # check resulting series and index
        expected_results = pd.Series([1, 2, 1, 0, 0,
                                      0, 1, 2, 0, 1], index=idx[10:])
        self.assertAlmostEqual(0, (ft_series - expected_results).abs().sum(), places=5)

    def test__feature_transform_oof_dummies(self):
        """ Test DummyTransformation OOF process. Values and Index are tested """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=True, random_state=None)
        # Call feature transformer
        ft = DummyTransformation(feature_name="test", drop_level=0, noise_level=None)
        ft_frame = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        full_frame = pd.concat([series, ft_frame], axis=1)
        # check resulting series and index
        expected_results = pd.read_csv(get_path("test_dummies_01.csv"), index_col=0)
        cols = ["test_A", "test_B", "test_C", "test_D"]
        self.assertAlmostEqual(0, (full_frame[cols] - expected_results[cols]).abs().sum().sum(), places=5)

    def test__feature_transform_oof_dummies_drop_nan(self):
        """ Test DummyTransformation with option to drop NaN values """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=True, random_state=None)
        # Call feature transformer
        ft = DummyTransformation(feature_name="test", drop_level=0, noise_level=None, keep_dum_cols_with_nan=False)
        ft_frame = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        full_frame = pd.concat([series, ft_frame], axis=1)
        # check resulting series and index
        expected_results = pd.read_csv(get_path("test_dummies_01.csv"), index_col=0)
        self.assertEqual(list(ft_frame.columns), ["test_A", "test_B", "test_C"])
        cols = ft_frame.columns
        self.assertAlmostEqual(0, (full_frame[cols] - expected_results[cols]).abs().sum().sum(), places=5)

    def test__feature_transform_oof_dummies_drop_level(self):
        """ Test DummyTransformation with drop_level values are drop below this frequency """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=True, random_state=None)
        # Call feature transformer
        ft = DummyTransformation(feature_name="test", drop_level=0.25, noise_level=None, keep_dum_cols_with_nan=True)
        ft_frame = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        full_frame = pd.concat([series, ft_frame], axis=1)
        # check resulting series and index
        expected_results = pd.read_csv(get_path("test_dummies_02.csv"), index_col=0)
        self.assertEqual(list(ft_frame.columns), ["test_A", "test_B", "test_other"])
        cols = ft_frame.columns
        self.assertAlmostEqual(0, (full_frame[cols] - expected_results[cols]).abs().sum().sum(), places=5)

    # def test__numerical_feature_transform_regressor(self):
    #     self.fail()

    def test__categorical_feature_transform_regressor_oof(self):
        """ Test CategoricalRegressorTransformation OOF process, categorical feature is moved to dummies
            and regression is performed (Here with a Lasso model """
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = CategoricalRegressorTransformation(feature_name="test",
                                                noise_level=0,
                                                regressor=Lasso(alpha=.01),
                                                drop_level=0.25,
                                                keep_dum_cols_with_nan=False)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        expected_res = [0.2375, 0.2375, 0.0500, 0.2375, 0.0500,
                        0.2375, 0.2375, 0.0500, 0.2375, 0.2375,
                        0.2750, 0.3333, 0.2750, 0.6333, 0.6333,
                        0.6333, 0.2750, 0.3333, 0.6333, 0.2750]

        self.assertAlmostEqual(0, (expected_res - ft_series).abs().mean(), places=4)

    def test__categorical_feature_transform_regressor_drop_level_problem(self):
        """ Test CategoricalRegressorTransformation OOF process, categorical feature is moved to dummies
            But here drop_level is not big enough train and validation sets do not have the same number of
            dummy columns. Therefore regressor cannot transform validation data and an exception is thrown """
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = CategoricalRegressorTransformation(feature_name="test",
                                                noise_level=0,
                                                regressor=Lasso(alpha=.01),
                                                drop_level=0,
                                                keep_dum_cols_with_nan=False)
        data_df = series.to_frame(name="test")
        for trn_idx, val_idx in folds.split(data_df, target):
            trn_X, trn_Y = data_df.iloc[trn_idx], target.iloc[trn_idx]
            val_X, val_Y = data_df.iloc[val_idx], target.iloc[val_idx]
            ft.fit(trn_X, trn_Y)
            self.assertRaises(ValueError, ft.transform, data=val_X)

    def test__categorical_feature_transform_regressor_no_oof(self):
        """ Test CategoricalRegressorTransformation fit / transform process, categorical feature is moved to dummies
                    and regression is performed (Here with a Lasso model """
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = CategoricalRegressorTransformation(feature_name="test",
                                                noise_level=0,
                                                regressor=Lasso(alpha=.01),
                                                drop_level=0.25,
                                                keep_dum_cols_with_nan=False)
        data_df = series.to_frame(name="test")
        expected_res = [0.2375, 0.2375, 0.0500, 0.2375, 0.0500,
                        0.2375, 0.2375, 0.0500, 0.2375, 0.2375,
                        0.2750, 0.3333, 0.2750, 0.6333, 0.6333,
                        0.6333, 0.2750, 0.3333, 0.6333, 0.2750]

        for i, (trn_idx, val_idx) in enumerate(folds.split(data_df, target)):
            trn_X, trn_Y = data_df.iloc[trn_idx], target.iloc[trn_idx]
            val_X, val_Y = data_df.iloc[val_idx], target.iloc[val_idx]
            ft.fit(trn_X, trn_Y)
            ft_series = ft.transform(val_X)
            self.assertAlmostEqual(0, (expected_res[i*10: i*10+10] - ft_series).abs().mean(), places=4)

    def test__categorical_feature_transform_classifier_oof(self):
        """ Test CategoricalClassifierTransformation OOF process, categorical feature is moved to dummies
            and classification is performed (Here with a Lasso model """
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = CategoricalClassifierTransformation(feature_name="test",
                                                 classifier=LogisticRegression(C=0.01),
                                                 probabilities=True,
                                                 noise_level=0,
                                                 drop_level=0.25,
                                                 keep_dum_cols_with_nan=False)
        ft_series = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        # ft_series.to_csv("test_classifier_proba.csv", index=True)
        expected_res = pd.read_csv(get_path("test_classifier_proba.csv"), index_col=0)

        self.assertAlmostEqual(0, (expected_res - ft_series).abs().mean().mean(), places=8)

    def test__categorical_feature_transform_classifier_no_oof(self):
        """ Test CategoricalClassifierTransformation fit/transform process, categorical feature is moved to dummies
            and classification is performed (Here with a Lasso model """
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = CategoricalClassifierTransformation(feature_name="test",
                                                 classifier=LogisticRegression(C=0.01),
                                                 probabilities=True,
                                                 noise_level=0,
                                                 drop_level=0.25,
                                                 keep_dum_cols_with_nan=False)
        data_df = series.to_frame(name="test")
        expected_res = pd.read_csv(get_path("test_classifier_proba.csv"), index_col=0)

        for i, (trn_idx, val_idx) in enumerate(folds.split(data_df, target)):
            trn_X, trn_Y = data_df.iloc[trn_idx], target.iloc[trn_idx]
            val_X, val_Y = data_df.iloc[val_idx], target.iloc[val_idx]
            ft.fit(trn_X, trn_Y)
            ft_series = ft.transform(val_X)
            self.assertAlmostEqual(0, (expected_res.iloc[i*10: i*10+10] - ft_series).abs().mean().mean(), places=8)

    def test_shadow_transformation(self):
        """ Test IdentityTransformation with shadow process."""
        np.random.seed(18)
        series = pd.Series(np.random.choice(3, 20, p=[0.40, 0.40, 0.20]))
        ft = ShadowTransformation(feature_name="test")
        np.random.seed(10)
        ft_series1 = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        ft = IdentityTransformation(feature_name="test", shadow=True)
        np.random.seed(10)
        ft_series2 = ft.fit_transform(data=pd.DataFrame(series, columns=["test"]))
        # ft_series = ft._feature_transform_same(series=series)
        self.assertAlmostEqual(0, np.sum(np.abs(ft_series1 - ft_series2)))
        self.assertNotAlmostEqual(0, np.sum(np.abs(series - ft_series1)))

    def test__feature_transform_oof_label_encoding_with_shadow(self):
        """ Test LabelEncodingTransformation with shadow process. Shadow process shuffles the output data.
            It is usefull for feature selection. Values and index are tested. """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series,
                                            p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = LabelEncodingTransformation(feature_name="test", noise_level=0, shadow=True)
        np.random.seed(10)
        ft_series1 = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        # check resulting series and index
        expected_list_1 = [1, 1, 2, 0, 2,
                           1, 1, -1, 0, 0]
        expected_list_2 = [1, 2, 1, 0, 0,
                           0, 1, 2, 0, 1]
        np.random.seed(10)
        np.random.shuffle(expected_list_1)
        np.random.shuffle(expected_list_2)
        expected_results = pd.Series(expected_list_1 + expected_list_2, index=idx)
        self.assertAlmostEqual(0, (ft_series1 - expected_results).abs().sum(), places=5)

    def test__feature_transform_oof_frequency_with_shadow(self):
        """ Test FrequencyTransformation with shadow process. Shadow process shuffles the output data.
            It is usefull for feature selection. Values and index are tested. """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice([1, 2, 4], len_series, p=[0.40, 0.40, 0.20]), name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = FrequencyTransformation(feature_name="test", noise_level=0, shadow=True)
        # check resulting series and index
        expected_list_1 = [0.4, 0.4, 0.2, 0.4, 0.2,
                                     0.4, 0.4, 0.2, 0.4, 0.4]
        expected_list_2 = [0.4, 0.3, 0.4, 0.3, 0.3,
                                      0.3, 0.4, 0.3, 0.3, 0.4]
        np.random.seed(10)
        ft_series1 = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)

        np.random.seed(10)
        np.random.shuffle(expected_list_1)
        np.random.shuffle(expected_list_2)
        expected_results = pd.Series(expected_list_1 + expected_list_2, index=idx)
        self.assertAlmostEqual(0, (ft_series1 - expected_results).abs().sum(), places=5)

    def test__feature_transform_oof_dummies_shadow(self):
        """ Test DummyTransformation OOF process with shadow shuffling. Values and Index are tested """
        # Create series and target
        len_series = 20
        np.random.seed(18)
        series = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], len_series, p=[0.40, 0.40, 0.15, 0.05]),
                           name='category')
        target = pd.Series(np.random.choice([0, 1], len_series, p=[0.7, 0.3]), name='target')
        # Create shuffled index
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        series.index = idx
        target.index = idx
        # Create folds
        folds = KFold(n_splits=2, shuffle=False, random_state=None)
        # Call feature transformer
        ft = DummyTransformation(feature_name="test", drop_level=0, noise_level=None, shadow=True)
        np.random.seed(10)
        ft_frame1 = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        shuff_frame = pd.concat([series, ft_frame1], axis=1)

        ft = DummyTransformation(feature_name="test", drop_level=0, noise_level=None, shadow=False)
        ft_frame2 = ft.get_oof_data(data=series.to_frame(name="test"), target=target, folds=folds)
        unshuff_frame = pd.concat([series, ft_frame2], axis=1)

        # First make sure index of shuffled and unshuffled results are equal
        self.assertEqual(True, shuff_frame["category"].equals(unshuff_frame["category"]))
        self.assertNotAlmostEqual(0, (ft_frame1 - ft_frame2).abs().sum().sum())

        # Now reproduce the shadow shuffling process
        np.random.seed(10)
        idx1 = np.arange(10)
        np.random.shuffle(idx1)
        idx2 = np.arange(10, 20)
        np.random.shuffle(idx2)

        # check things are as they should be
        self.assertAlmostEqual(0, np.sum(np.sum(np.abs(
            np.vstack((ft_frame2.loc[ft_frame2.index[idx1]].values,
                       ft_frame2.loc[ft_frame2.index[idx2]].values)) - ft_frame1.values))))