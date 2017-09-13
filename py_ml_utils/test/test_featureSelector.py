from unittest import TestCase
from py_ml_utils.feature_transformer import *
from py_ml_utils.feature_selector import *
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from itertools import combinations


class TestFeatureTransformer(TestCase):

    def test_feature_selection_with_importances_log_loss(self):
        """ Test FeatureSelector with the iris dataset and log_loss metric
            This also checks feature importance works """
        iris_dataset = datasets.load_iris()

        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=3)
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        dataset = pd.DataFrame(iris_dataset.data, columns=features)
        for f1, f2 in combinations(features, 2):
            dataset[f1 + "_/_" + f2] = dataset[f1] / dataset[f2]

        for f1, f2 in combinations(features, 2):
            dataset[f1 + "_*_" + f2] = dataset[f1] * dataset[f2]

        tf_pairs = []
        for f_ in dataset.columns:
            tf_pairs.append(
                FeatureTransformationPair(
                    transformer=IdentityTransformation(feature_name=f_),
                    missing_inferer=None
                )
            )
            tf_pairs.append(
                FeatureTransformationPair(
                    transformer=ShadowTransformation(feature_name=f_),
                    missing_inferer=None
                )
            )

        fs = FeatureSelector(max_features=.5, max_runs=20)

        clf = RandomForestClassifier(n_estimators=50,
                                     max_depth=7,
                                     max_features=.2,
                                     criterion="entropy",
                                     random_state=None,
                                     n_jobs=-1)
        np.random.seed(10)
        scores = fs.select(dataset=dataset,
                           target=pd.Series(iris_dataset.target, name="iris_class"),
                           pairs=tf_pairs,
                           estimator=clf,
                           metric=log_loss,
                           probability=True,
                           folds=folds,
                           maximize=False)

        scores.sort_values(by="importance", ascending=False, inplace=True)

        self.assertAlmostEqual(0.084482, scores.importance.values[0], places=6)
        self.assertEqual("petal_length_*_petal_width", scores.feature.values[0])
        self.assertAlmostEqual(1.0, scores.importance.sum())

    def test_feature_selection_without_importances_log_loss(self):
        """ Test FeatureSelector with the iris dataset and log_loss metric with a classifier that does not
            provide feature_importances """
        iris_dataset = datasets.load_iris()

        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=3)
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        dataset = pd.DataFrame(iris_dataset.data, columns=features)
        for f1, f2 in combinations(features, 2):
            dataset[f1 + "_/_" + f2] = dataset[f1] / dataset[f2]

        for f1, f2 in combinations(features, 2):
            dataset[f1 + "_*_" + f2] = dataset[f1] * dataset[f2]

        tf_pairs = []
        for f_ in dataset.columns:
            tf_pairs.append(
                FeatureTransformationPair(
                    transformer=IdentityTransformation(feature_name=f_),
                    missing_inferer=None
                )
            )
            tf_pairs.append(
                FeatureTransformationPair(
                    transformer=ShadowTransformation(feature_name=f_),
                    missing_inferer=None
                )
            )

        fs = FeatureSelector(max_features=.5, max_runs=100)

        clf = LogisticRegression()

        np.random.seed(10)
        scores = fs.select(dataset=dataset,
                           target=pd.Series(iris_dataset.target, name="iris_class"),
                           pairs=tf_pairs,
                           estimator=clf,
                           metric=log_loss,
                           probability=True,
                           folds=folds,
                           maximize=False)

        # scores.sort_values(by="score", ascending=True, inplace=True)

        print(scores.sort_values(by="score", ascending=True).head())

        self.assertAlmostEqual(0.13078, scores.score.values[0], places=5)
        self.assertEqual("petal_length_*_petal_width", scores.feature.values[0])

    def test_feature_selection_without_importances_accuracy(self):
        """ Test feature selector with iris dataset and accuracy metric
            This also checks maximize and probability arguments """
        iris_dataset = datasets.load_iris()

        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=3)
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        dataset = pd.DataFrame(iris_dataset.data, columns=features)
        for f1, f2 in combinations(features, 2):
            dataset[f1 + "_/_" + f2] = dataset[f1] / dataset[f2]

        for f1, f2 in combinations(features, 2):
            dataset[f1 + "_*_" + f2] = dataset[f1] * dataset[f2]

        tf_pairs = []
        for f_ in dataset.columns:
            tf_pairs.append(
                FeatureTransformationPair(
                    transformer=IdentityTransformation(feature_name=f_),
                    missing_inferer=None
                )
            )
            tf_pairs.append(
                FeatureTransformationPair(
                    transformer=ShadowTransformation(feature_name=f_),
                    missing_inferer=None
                )
            )

        fs = FeatureSelector(max_features=.5, max_runs=100)

        clf = LogisticRegression()

        np.random.seed(10)
        scores = fs.select(dataset=dataset,
                           target=pd.Series(iris_dataset.target, name="iris_class"),
                           pairs=tf_pairs,
                           estimator=clf,
                           metric=accuracy_score,
                           probability=False,
                           folds=folds,
                           maximize=True)
        # scores are expected to be sorted in descending order using the score column
        self.assertAlmostEqual(0.951818, scores.score.values[0], places=6)
        self.assertEqual("petal_length_*_petal_width", scores.feature.values[0])
