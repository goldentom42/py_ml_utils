from unittest import TestCase
import pandas as pd
import numpy as np
from py_ml_utils.missing_value_inferer import *


class TestMissingValueInferer(TestCase):

    def test_infer_missing_value_constant(self):
        """ Test MeanMissingValueInferer, replacing np.nan by mean """
        idx = np.arange(12)
        np.random.shuffle(idx)
        series = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, np.nan, np.nan], index=idx)
        mvi = ConstantMissingValueInferer(feature_name="test", missing_value=np.nan, replacement=-1)
        ft_series = mvi.infer(series.to_frame(name="test"))
        self.assertEqual(ft_series.values[10], -1)
        self.assertEqual(ft_series.values[11], -1)
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series.index) - np.array(ft_series.index))))

    def test_infer_missing_value_mean(self):
        """ Test MeanMissingValueInferer, replacing np.nan by mean """
        idx = np.arange(12)
        np.random.shuffle(idx)
        series = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, np.nan, np.nan], index=idx)
        mvi = MeanMissingValueInferer(feature_name="test", missing_value=np.nan)
        ft_series = mvi.infer(series.to_frame(name="test"))
        self.assertEqual(ft_series.values[10], 0.6)
        self.assertEqual(ft_series.values[11], 0.6)
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series.index) - np.array(ft_series.index))))

    def test_infer_missing_value_mean_index(self):
        """ Test MeanMissingValueInferer, check returned series has same index as input Series """
        idx = np.arange(12)
        np.random.shuffle(idx)
        series = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, np.nan, np.nan], index=idx)
        idx = np.arange(12)
        np.random.shuffle(idx)
        series.index = idx
        mvi = MeanMissingValueInferer(feature_name="test", missing_value=np.nan)
        ft_series = mvi.infer(series.to_frame(name="test"))
        self.assertEqual(0, np.mean(np.abs((ft_series.index - series.index))))
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series.index) - np.array(ft_series.index))))

    def test_infer_missing_value_median(self):
        """ Test MedianMissingValueInferer, replacing np.nan by median """
        idx = np.arange(12)
        np.random.shuffle(idx)
        series = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, np.nan, np.nan], index=idx)
        mvi = MedianMissingValueInferer(feature_name="test", missing_value=np.nan)
        ft_series = mvi.infer(series.to_frame(name="test"))
        self.assertEqual(ft_series.values[10], 1.0)
        self.assertEqual(ft_series.values[11], 1.0)
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series.index) - np.array(ft_series.index))))

    def test_infer_missing_value_most_frequent(self):
        """ Test MostFrequentMissingValueInferer, replacing np.nan by most frequent value """
        idx = np.arange(12)
        np.random.shuffle(idx)
        series = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, np.nan, np.nan], index=idx)
        mvi = MostFrequentMissingValueInferer(feature_name="test", missing_value=np.nan)
        ft_series = mvi.infer(series.to_frame(name="test"))
        self.assertEqual(ft_series.values[10], 1)
        self.assertEqual(ft_series.values[11], 1)
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series.index) - np.array(ft_series.index))))

    def test_infer_missing_value_most_frequent_missing_value_arg(self):
        """ Test MostFrequentMissingValueInferer, replacing -1 by most frequent value """
        idx = np.arange(12)
        np.random.shuffle(idx)
        series = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, -1, -1.0], index=idx)
        mvi = MostFrequentMissingValueInferer(feature_name="test", missing_value=-1)
        ft_series = mvi.infer(series.to_frame(name="test"))
        self.assertEqual(ft_series.values[10], 1)
        self.assertEqual(ft_series.values[11], 1)
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series.index) - np.array(ft_series.index))))

    def test_infer_missing_value_no_series(self):
        """ Test exception when no Series provided """
        mvi = MissingValueInferer()
        self.assertRaises(ValueError, mvi.infer, None)

    def test_infer_missing_value_empty_series(self):
        """ Test exception when empty Series provided """
        mvi = MissingValueInferer()
        self.assertRaises(ValueError, mvi.infer, dataset=pd.DataFrame())

    def test_infer_missing_value_using_groupby(self):
        """ Test GroupByMissingValueInferer, where missing value is infered using other features in the dataset """
        len_series = 30
        idx = np.arange(len_series)
        np.random.shuffle(idx)
        np.random.seed(18)
        f1_series = pd.Series(np.random.choice(['A', 'B', 'C'], len_series, p=[0.40, 0.30, 0.30]), name='f1', index=idx)
        f2_series = pd.Series(np.random.choice(['D', 'E'], len_series, p=[0.40, 0.60]), name='f2', index=idx)
        target = pd.Series(np.random.choice([0, 1, np.nan], len_series, p=[0.6, 0.3, 0.1]), name='target', index=idx)

        mvi = GroupByMissingValueInferer(feature_name="target",
                                         missing_value=np.nan,
                                         groupby=["f1", "f2"],
                                         average_type="MEAN")
        series1 = mvi.infer(dataset=pd.concat([f1_series, f2_series, target], axis=1))
        series2 = mvi.infer(dataset=[pd.concat([f1_series, f2_series, target], axis=1)])

        expected_result = pd.Series([1.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0,
                           0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0], index=idx)

        self.assertAlmostEqual(0, (series1 - expected_result).abs().mean())
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series1.index) - np.array(expected_result.index))))
        self.assertAlmostEqual(0, (series2 - expected_result).abs().mean())
        self.assertAlmostEqual(0, np.sum(np.abs(np.array(series2.index) - np.array(expected_result.index))))

        series1 = mvi.infer(dataset=[pd.concat([f1_series[:20], f2_series[:20], target[:20]], axis=1),
                                     pd.concat([f1_series[20:], f2_series[20:], target[20:]], axis=1)])
        expected0 = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        expected1 = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        self.assertAlmostEqual(0, (series1[0] - expected0).abs().mean())
        self.assertAlmostEqual(0, (series1[1] - expected1).abs().mean())

    def test_infer_missing_value_using_groupby_index(self):
        """ Test GroupByMissingValueInferer, where missing value is infered using other features in the dataset
            Here we check that output Serie index is equal to inpu Series index """
        len_series = 30
        np.random.seed(18)
        f1_series = pd.Series(np.random.choice(['A', 'B', 'C'], len_series, p=[0.40, 0.30, 0.30]), name='f1')
        f2_series = pd.Series(np.random.choice(['D', 'E'], len_series, p=[0.40, 0.60]), name='f2')
        target = pd.Series(np.random.choice([0, 1, np.nan], len_series, p=[0.6, 0.3, 0.1]), name='target')

        idx = np.arange(len_series)
        np.random.shuffle(idx)
        f1_series.index = idx
        f2_series.index = idx
        target.index = idx

        mvi = GroupByMissingValueInferer(feature_name="target", missing_value=np.nan, groupby=["f1", "f2"],
                                         average_type="MEAN")
        series1 = mvi.infer(dataset=pd.concat([f1_series, f2_series, target], axis=1))

        expected_result = [1.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0,
                           0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                           0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        self.assertAlmostEqual(0, (series1 - expected_result).abs().mean())
        self.assertAlmostEqual(0, np.mean(np.abs((series1.index - target.index))))
