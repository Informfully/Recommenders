#############################################################################################################                                                             #
# NAME:         ratio_split_percent.py                                                                      #                                                      #
# DATE:         2024-11-13                                                                                  #
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  use different data sets to form a combination critiron for radio split                      #
#############################################################################################################

from math import ceil

from ..utils import get_rng
from ..utils.common import safe_indexing
import numpy as np
import pandas as pd
from .ratio_split import RatioSplit


class RatioSplitPercent(RatioSplit):
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
    def __init__(self, data, data2=None, data3=None, data4=None, 
                 group_by="sentiment",
                 top_percentage=100.0, 
                 test_size=0.2, val_size=0.0, rating_threshold=1.0,
                 seed=None, exclude_unknowns=True, verbose=False, 
                 ascending=False, 
                 save_test_data=False, test_data_path='test_data.csv',
                 **kwargs):
        super().__init__(data=data, rating_threshold=rating_threshold, seed=seed,
                         exclude_unknowns=exclude_unknowns, verbose=verbose, **kwargs)

        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.group_by = group_by
        self.top_percentage = top_percentage
        self.ascending = ascending
        self.save_test_data = save_test_data
        self.test_data_path = test_data_path
        self.train_size, self.val_size, self.test_size = self.validate_size(val_size, test_size, len(self._data))
        self._split()
    
    @staticmethod
    def validate_size(val_size, test_size, num_ratings):
        if val_size is None:
            val_size = 0.0
        elif val_size < 0:
            raise ValueError('val_size={} should be greater than zero'.format(val_size))
        elif val_size >= num_ratings:
            raise ValueError(
                'val_size={} should be less than the number of ratings {}'.format(val_size, num_ratings))

        if test_size is None:
            test_size = 0.0
        elif test_size < 0:
            raise ValueError('test_size={} should be greater than zero'.format(test_size))
        elif test_size >= num_ratings:
            raise ValueError(
                'test_size={} should be less than the number of ratings {}'.format(test_size, num_ratings))

        if val_size < 1:
            val_size = ceil(val_size * num_ratings)
        if test_size < 1:
            test_size = ceil(test_size * num_ratings)

        if val_size + test_size >= num_ratings:
            raise ValueError(
                'The sum of val_size and test_size ({}) should be smaller than the number of ratings {}'.format(
                    val_size + test_size, num_ratings))

        train_size = num_ratings - (val_size + test_size)

        return int(train_size), int(val_size), int(test_size)

    def _split(self):
        # choose the data to group by
        if self.group_by == "sentiment":
            attribute_data = self.data2
        elif self.group_by == "complexity":
            attribute_data = self.data3
        elif self.group_by == "category":
            attribute_data = self.data4
        else:
            raise ValueError("group_by option must be 'sentiment', 'complexity', or 'category'")

        if attribute_data is None:
            raise ValueError(f"No data provided for group_by = '{self.group_by}'")

        #create a list of tuples with the index and the attribute value
        indexed_data = [(idx, attribute_data.get(user_id)) for idx, (user_id, item_id, rating) in enumerate(self._data)]
        indexed_data = [x for x in indexed_data if x[1] is not None]  # Remove None values

        # calculate and print the average rating for all data
        all_ratings = [self._data[idx][2] for idx in range(len(self._data))]  
        mean_all_ratings = np.mean(all_ratings)
        print(f"100% data's average rating score: {mean_all_ratings}")

        # sort the data by the attribute value
        indexed_data.sort(key=lambda x: x[1], reverse=not self.ascending)

        # kepp only the top percentage of the data
        num_to_keep = int(len(indexed_data) * (self.top_percentage / 100.0))
        top_indices = [idx for idx, _ in indexed_data[:num_to_keep]]
        print("Top sorted data:", top_indices[:10])
        print("Bottom sorted data:", top_indices[-10:])

        # Calculate and print the average rating score for the top percentage of the data
        top_ratings = [self._data[idx][2] for idx in top_indices]  
        mean_top_ratings = np.mean(top_ratings)
        print(f"{self.top_percentage}% data's average rating score: {mean_top_ratings}")

        # Randomly shuffle the data
        self.rng.shuffle(top_indices)
        train_idx = top_indices[:self.train_size]
        test_idx = top_indices[-self.test_size:]
        val_idx = top_indices[self.train_size:-self.test_size] if self.val_size > 0 else []

        # Extract the data
        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)
        val_data = safe_indexing(self._data, val_idx) if len(val_idx) > 0 else None

        # Save the test data
        if self.save_test_data:
            self._save_test_data(test_data)

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)

    def _save_test_data(self, test_data):
        test_df = pd.DataFrame(test_data, columns=['user_id', 'item_id', 'rating'])
        test_df.to_csv(self.test_data_path, index=False)
        print(f"Test data saved to '{self.test_data_path}'")



