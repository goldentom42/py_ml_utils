import numpy as np
import pandas as pd


class MissingValueInferer(object):

    _process_name=None

    def __init__(self, feature_name=None, missing_value=np.nan):
        self.name = feature_name
        self.missing_value = missing_value
        self.series = None

    @property
    def process_name(self):
        return self._process_name

    def _infer_specific_process(self):
        raise NotImplementedError()

    def infer(self, dataset):
        if (dataset is None) or (len(dataset) == 0):
            raise ValueError("No series provided, input is None or empty.")
        self.series = dataset[self.name].replace(self.missing_value, np.nan)
        return self._infer_specific_process()


class ConstantMissingValueInferer(MissingValueInferer):
    _process_name = "Constant"

    def __init__(self, feature_name=None, missing_value=np.nan, replacement=-1):
        # Call super
        super(ConstantMissingValueInferer, self).__init__(feature_name, missing_value)
        self.replacer = replacement

    def _infer_specific_process(self):
        return self.series.fillna(self.replacer)


class MeanMissingValueInferer(MissingValueInferer):
    _process_name = "MeanInferer"

    def _infer_specific_process(self):
        the_mean = self.series.mean()
        return self.series.fillna(the_mean)


class MedianMissingValueInferer(MissingValueInferer):
    _process_name = "MedianInferer"

    def _infer_specific_process(self):
        the_mean = self.series.median()
        return self.series.fillna(the_mean)


class MostFrequentMissingValueInferer(MissingValueInferer):
    _process_name = "MostFrequentInferer"

    def _infer_specific_process(self):
        most_frequent = self.series.value_counts().index[0]
        return self.series.fillna(most_frequent)


class GroupByMissingValueInferer(MissingValueInferer):
    _process_name = "GroupByInferer"

    def __init__(self,
                 feature_name=None,
                 missing_value=np.nan,
                 groupby=None,
                 average_type='MEAN'):
        # Call super
        super(GroupByMissingValueInferer, self).__init__(feature_name, missing_value)
        self.groupby = groupby
        self.average_type = average_type

    def infer(self, dataset=None):
        """
                Replace NaN values of *feature** by its average values against provided *groupby* directive
                This method does not mutate input parameters
                :param dataset: list of dataframes or 1 dataframe used to compute averages and whose *feature* NaN values will be replaced
                :param feature: name of the feature whose missing values have to be replaced
                :param groupby: list of features used to compute averages
                :param average_type: can be 'MEAN' of 'MEDIAN', default is 'MEAN'
                :return: list of feature pd.Series whose NaN are replaced
                """
        # First concatenate datasets
        if type(dataset) is not list:
            datasets = [dataset]
        else:
            datasets = dataset

        # Drop index to ensure no conflict occurs due to indexing
        full_dataset = datasets[0][self.groupby + [self.name]].reset_index(drop=True).copy()
        full_dataset[self.name].replace(self.missing_value, np.nan, inplace=True)

        for i in range(1, len(datasets)):
            full_dataset = pd.concat([full_dataset,
                                      datasets[0][self.groupby + [self.name]].reset_index(drop=True).copy()],
                                     axis=0)

        # use Groupby to compute averages. The mean method excludes NaN
        if self.average_type == 'MEAN':
            averages = full_dataset.groupby(by=self.groupby).mean().reset_index()
            full_average = full_dataset[self.name].mean()
        elif self.average_type == 'MEDIAN':
            averages = full_dataset.groupby(by=self.groupby).median().reset_index()
            full_average = full_dataset[self.name].median()
        else:
            raise ValueError("Unsupported average type " + str(self.average_type) + ". Can be 'MEAN' or 'MEDIAN'.")

        # del full dataset since it was just a temporary dataframe used to compute averages
        del full_dataset

        # print(datasets)

        # Create reduced datasets for merging
        reduced_datasets = []
        for i, dataset in enumerate(datasets):
            reduced_datasets.append(dataset.loc[:, self.groupby + [self.name]])
            reduced_datasets[i][self.name].replace(self.missing_value, np.nan, inplace=True)

        # Now we have an averages DataFrame containing groupby features and the feature averages
        # create a new groupby column which is the concatenation of the groupby features
        # in both groupby averages and extracted datasets
        averages['groupby_feat'] = ""
        for f in self.groupby:
            averages['groupby_feat'] += "_"
            averages['groupby_feat'] += averages[f].astype(str)

        for dataset in reduced_datasets:
            dataset['groupby_feat'] = ""
            for f in self.groupby:
                dataset['groupby_feat'] += "_"
                dataset['groupby_feat'] += dataset[f].astype(str)

        # Merge extracted datasets and groupby averages
        merged_datasets = []
        for i, dataset in enumerate(reduced_datasets):
            # Merge dataset with averages
            merged_datasets.append(pd.merge(left=dataset,
                                            right=averages,
                                            how='left',
                                            on="groupby_feat",
                                            suffixes=["", "_avg"]))
            # merge removes indexing so set index
            merged_datasets[i].index = dataset.index

        # Replace NaN by averages
        for dataset in merged_datasets:
            dataset[self.name].fillna(dataset[self.name + "_avg"], inplace=True)
            dataset[self.name].fillna(full_average, inplace=True)

        # Return list of Series corresponding to the datasets
        returned_data = [dataset[self.name] for dataset in merged_datasets]
        if len(returned_data) == 1:
            return returned_data[0]
        else:
            return returned_data
