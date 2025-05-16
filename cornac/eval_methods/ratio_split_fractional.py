#############################################################################################################
# NAME:         ratio_split_fractional.py                                                                   #                                                     #
# DATE:         2024-11-13                                                                                  #
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  select a part of data set to make ratio split                                               #
#############################################################################################################

from math import ceil

from ..utils.common import safe_indexing
from .ratio_split import RatioSplit


class RatioSplitFractional(RatioSplit):
    """
    Splitting data into training, validation, and test sets based on provided sizes.
    Data is always shuffled before split.

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    test_size: float, optional, default: 0.2
        The proportion of the test set, \
        if > 1 then it is treated as the size of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    data_percentage: float, optional, default: 100.0
        Percentage of the dataset to use, must be between 0 and 100.
    """

    def __init__(self, data, test_size=0.2, val_size=0.0, rating_threshold=1.0,
                 seed=None, exclude_unknowns=True, verbose=False, data_percentage=100.0, 
                 save_test_data=False, test_data_path='test_data.csv',**kwargs):
        
        super().__init__(
            data=data,
            test_size=test_size,
            val_size=val_size,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
        )

        self.data_percentage = self.validate_percentage(data_percentage)
        self.data = self.select_percentage(self.data, self.data_percentage)
        
        self.train_size, self.val_size, self.test_size = self.validate_size(val_size, test_size, len(self.data))
        self.save_test_data = save_test_data
        self.test_data_path = test_data_path
        self._split()

    @staticmethod
    def validate_percentage(data_percentage):
        if not (0 < data_percentage <= 100):
            raise ValueError("data_percentage={} should be between 0 and 100".format(data_percentage))
        return data_percentage

    def select_percentage(self, data, percentage):
        # Randomly select the specified percentage of data
        num_samples = ceil(len(data) * (percentage / 100.0))
        data_idx = self.rng.permutation(len(data))[:num_samples]
        return safe_indexing(data, data_idx)
    