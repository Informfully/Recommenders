#############################################################################################################
#
# NAME:         stratified_split_diversity.py                                                                      #                                                        #
# DATE:         2024-11-21                                                                                  #
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  use different splitting methods to train the model for diversity improvement                 #
#######
import pandas as pd
import numpy as np
import os
from math import ceil
from collections import defaultdict

from .base_method import BaseMethod
from .ratio_split import RatioSplit
from ..utils import get_rng
from ..utils.common import safe_indexing


class StratifiedSplitDiv(BaseMethod):
    """Grouping data by user or item then splitting data into training, validation, and test sets.

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value, timestamp)].

    group_by: str, optional, default: 'user'
        Grouping by 'user' or 'item'.

    chrono: bool, optional, default False
        Data is ordered by reviewed time or not. If this option is True, data must be in 'UIRT' format.

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

    """

    def __init__(
        self,
        data,
        group_by="user",
        chrono=False,
        fmt="UIR",
        test_size=0.2,
        val_size=0.0,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        super().__init__(
            data=data,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs
        )

        if group_by not in ["user", "item"]:
            raise ValueError(
                "group_by option must be either 'user' or 'item' but {}".format(
                    group_by
                )
            )

        if chrono and (fmt != "UIRT" or len(self._data[0]) != 4):
            raise ValueError(
                'Input data must be in "UIRT" format for sorting chronologically.'
            )

        self.chrono = chrono
        self.group_by = group_by
        self.val_size = val_size
        self.test_size = test_size

        self._split()

    def _split(self):
        data = (
            sorted(self._data, key=lambda x: x[3]) if self.chrono else self._data
        )  # sort data chronologically

        grouped_indices = defaultdict(list)
        for idx, (uid, iid, *_) in enumerate(data):
            if self.group_by == "user":
                grouped_indices[uid].append(idx)
            else:
                grouped_indices[iid].append(idx)

        train_idx = []
        test_idx = []
        val_idx = []

        for rating_indices in grouped_indices.values():
            n_ratings = len(rating_indices)
            n_train, _, n_test = RatioSplit.validate_size(
                self.val_size, self.test_size, n_ratings
            )

            if self.chrono:
                # training portion is chronologically sorted
                # validation and test portions are randomly selected
                rating_indices = (
                    rating_indices[:n_train]
                    + self.rng.permutation(rating_indices[n_train:]).tolist()
                )
            else:
                rating_indices = self.rng.permutation(rating_indices).tolist()

            train_idx += rating_indices[:n_train]
            test_idx += rating_indices[-n_test:]
            val_idx += rating_indices[n_train:-n_test]

        train_data = safe_indexing(data, train_idx)
        test_data = safe_indexing(data, test_idx)
        val_data = safe_indexing(data, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)

from collections import defaultdict
import pandas as pd
import numpy as np

class StratifiedAttributeSplit(BaseMethod):
    def __init__(
        self,
        data1,
        data2=None,
        data3=None,
        group_by="category",
        fmt="UIRT",
        train_size=0.8,
        test_size=0.2,
        val_size=0.0,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        super().__init__(
            data=data1,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs
        )
        self.data1 = data1  
        self.data2 = data2 
        self.data3 = data3
        self.group_by = group_by
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

        if group_by not in ["category", "rating", "sentiment", "complexity"]:
            raise ValueError(
                f"group_by option must be one of 'category', 'rating', 'sentiment', or 'complexity', but got '{group_by}'"
            )

        # Determine if 'category' column exists in data1
        self.has_category = len(data1[0]) == 4 if data1 else False

        # Validate presence of 'category' when group_by is 'category'
        if group_by == "category" and not self.has_category:
            raise ValueError("Cannot group by 'category' because 'category' column is missing in data1.")

        self.split()

    def split(self):
        data1 = self.data1
        data2 = self.data2
        data3 = self.data3

        # Prepare intervals for complexity and sentiment if needed
        if self.group_by in ["complexity", "sentiment"]:
            if self.group_by == "complexity" and data2:
                min_complexity = min(data2.values())
                max_complexity = max(data2.values())
                complexity_range = max_complexity - min_complexity
                complexity_interval = complexity_range / 20 if complexity_range != 0 else 1
            elif self.group_by == "sentiment" and data3:
                min_sentiment = min(data3.values())
                max_sentiment = max(data3.values())
                sentiment_range = max_sentiment - min_sentiment
                sentiment_interval = sentiment_range / 20 if sentiment_range != 0 else 1
            else:
                raise ValueError(f"Data for '{self.group_by}' is missing.")

        grouped_indices = defaultdict(list)
        for idx, record in enumerate(data1):
            if self.has_category:
                uid, iid, rating, category = record
            else:
                uid, iid, rating = record

            if self.group_by == "category":
                if pd.isnull(category):
                    continue
                grouped_indices[category].append(idx)

            elif self.group_by == "rating":
                if pd.isnull(rating):
                    continue
                grouped_indices[rating].append(idx)

            elif self.group_by == "complexity":
                complexity = data2.get(iid)
                if complexity is None or pd.isnull(complexity):
                    continue
                complexity_bin = int((complexity - min_complexity) / complexity_interval)
                grouped_indices[complexity_bin].append(idx)

            elif self.group_by == "sentiment":
                sentiment = data3.get(iid)
                if sentiment is None or pd.isnull(sentiment):
                    continue
                sentiment_bin = int((sentiment - min_sentiment) / sentiment_interval)
                grouped_indices[sentiment_bin].append(idx)

        train_idx, val_idx, test_idx = [], [], []
        rng = np.random.default_rng(self.seed)

        for group, indices in grouped_indices.items():
            n_items = len(indices)
            if n_items < 10:
                train_idx.extend(indices)
            else:
                n_train = int(n_items * self.train_size)
                n_test = int(n_items * self.test_size)
                n_val = n_items - n_train - n_test
                if n_train == 0 and n_test == 0:
                    train_idx.extend(indices)
                else:
                    shuffled_indices = rng.permutation(indices)
                    train_idx.extend(shuffled_indices[:n_train])
                    val_idx.extend(shuffled_indices[n_train:n_train + n_val])
                    test_idx.extend(shuffled_indices[n_train + n_val:])

        train_data = [data1[i] for i in train_idx]
        val_data = [data1[i] for i in val_idx] if val_idx else None
        test_data = [data1[i] for i in test_idx]

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
        self.print_statistics(train_data, test_data, val_data)

    def print_statistics(self, train_data, test_data, val_data):
        def stats(data, name):
            num_users = len(set(row[0] for row in data)) if data else 0
            num_items = len(set(row[1] for row in data)) if data else 0
            num_ratings = len(data) if data else 0
            ratings = [row[2] for row in data] if data else []
            max_rating = max(ratings) if ratings else None
            min_rating = min(ratings) if ratings else None
            mean_rating = np.mean(ratings) if ratings else None

            print(f"{name} data:")
            print(f"  Number of users: {num_users}")
            print(f"  Number of items: {num_items}")
            print(f"  Number of ratings: {num_ratings}")
            print(f"  Max rating: {max_rating}")
            print(f"  Min rating: {min_rating}")
            print(f"  Mean rating: {mean_rating:.4f}" if mean_rating is not None else "  Mean rating: N/A")
            print("---")

        stats(train_data, "Training")
        stats(test_data, "Test")
        if val_data:
            stats(val_data, "Validation")

class SortedSplit(BaseMethod):
    def __init__(
        self,
        data1,
        data2=None,
        data3=None,
        group_by="complexity",
        fmt="UIRT",
        train_size=0.8,
        test_size=0.2,
        val_size=0.0,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        output_test_csv=None,
        **kwargs
    ):
        super().__init__(
            data=data1,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            output_test_csv=None,
            **kwargs
        )
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.output_test_csv = output_test_csv
        
        if group_by not in ["sentiment", "complexity"]:
            raise ValueError(
                "group_by option must be either 'sentiment' or 'complexity' but {}".format(group_by)
            )
        self.group_by = group_by
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        
        # Check if data1 has a category column
        self.has_category = len(data1[0]) == 4 if data1 else False

        self.split()

    def split(self):
        data1 = self.data1
        data2 = self.data2
        data3 = self.data3
        
        grouped_indices = defaultdict(list)
        for idx, record in enumerate(data1):
            if self.has_category:
                uid, iid, rating, category = record
            else:
                uid, iid, rating = record[:3]
                category = None

            if self.group_by == "complexity":
                complexity = data2.get(iid)
                if pd.isnull(complexity) or (self.has_category and pd.isnull(category)):
                    continue
                grouped_indices['complexity'].append((idx, complexity))
            elif self.group_by == "sentiment":
                sentiment = data3.get(iid)
                if pd.isnull(sentiment) or (self.has_category and pd.isnull(category)):
                    continue
                grouped_indices['sentiment'].append((idx, sentiment))
            else:
                continue
            
        train_idx, val_idx, test_idx = [], [], []
        train_values, test_values = [], []
        
        for key, group in grouped_indices.items():
            group.sort(key=lambda x: x[1], reverse=True)
            sorted_indices = [idx for idx, value in group]
            sorted_values = [value for idx, value in group]
            n_train = int(self.train_size * len(sorted_indices))
            train_idx += sorted_indices[:n_train]
            train_values += sorted_values[:n_train]
            test_idx += sorted_indices[n_train:]
            test_values += sorted_values[n_train:]

        train_data = safe_indexing(data1, train_idx)
        test_data = safe_indexing(data1, test_idx)
        val_data = safe_indexing(data1, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)

        if self.group_by in ["complexity", "sentiment"]:
            train_mean = np.mean(train_values)
            test_mean = np.mean(test_values)
            print(f"Training {self.group_by} mean: {train_mean}")
            print(f"Testing {self.group_by} mean: {test_mean}")

        num_train_users = len(set([row[0] for row in train_data]))
        num_train_items = len(set([row[1] for row in train_data]))
        num_train_ratings = len(train_data)
        max_train_rating = max([row[2] for row in train_data])
        min_train_rating = min([row[2] for row in train_data])
        mean_train_rating = np.mean([row[2] for row in train_data])

        num_test_users = len(set([row[0] for row in test_data]))
        num_test_items = len(set([row[1] for row in test_data]))
        num_test_ratings = len(test_data)
        max_test_rating = max([row[2] for row in test_data])
        min_test_rating = min([row[2] for row in test_data])
        mean_test_rating = np.mean([row[2] for row in test_data])
        print(f"Training data:")
        print(f"Number of users = {num_train_users}")
        print(f"Number of items = {num_train_items}")
        print(f"Number of ratings = {num_train_ratings}")
        print(f"Max rating = {max_train_rating}")
        print(f"Min rating = {min_train_rating}")
        print(f"Global mean = {mean_train_rating:.4f}")
        print(f"---")
        print(f"Test data:")
        print(f"Number of users = {num_test_users}")
        print(f"Number of items = {num_test_items}")
        print(f"Number of ratings = {num_test_ratings}")
        print(f"Max rating = {max_test_rating}")
        print(f"Min rating = {min_test_rating}")
        print(f"Global mean = {mean_test_rating:.4f}")

        if self.output_test_csv:
            output_dir = os.path.dirname(self.output_test_csv)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            columns = ['user', 'item', 'rating'] + (['category'] if self.has_category else [])
            test_df = pd.DataFrame(test_data, columns=columns)
            test_df.to_csv(self.output_test_csv, index=False)
            print(f"Testing data saved to {self.output_test_csv}")

class StratifiedDiversitySplit(BaseMethod):
    def __init__(
        self,
        data1,
        data2=None,
        data3=None,
        data4=None,
        group_by="sentiment",
        fmt="UIR",
        train_size=0.8,
        test_size=0.2,
        val_size=0.0,
        seed=None,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
        output_test_csv=None,
        num_bins=10, # number of levels for diversity scores
        **kwargs
    ):
        super().__init__(
            data=data1,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            output_test_csv=None,
            **kwargs
        )
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.group_by = group_by
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.output_test_csv = output_test_csv
        self.num_bins = num_bins
        if group_by not in ["category", "complexity", "sentiment"]:
            raise ValueError("group_by option must be 'category', 'complexity', or 'sentiment'")
        self.has_category = len(data1[0]) == 4 if data1 else False

        self.split()

    def split(self):
        print(f"Initial data sizes - data1: {len(self.data1)}, data2: {len(self.data2 if self.data2 else [])}, data3: {len(self.data3 if self.data3 else [])}")
        data1 = self.data1
        data2 = self.data2
        data3 = self.data3
        data4 = self.data4
    
    # Choose the attribute data to group by
        if self.group_by == "sentiment":
            attribute_data = data2
        elif self.group_by == "complexity":
            attribute_data = data3
        elif self.group_by == "category":
            attribute_data = data4
    
        # Create a dictionary to store the indices of the data points
        grouped_indices = defaultdict(list)
        for idx, record in enumerate(data1):
            uid, iid, rating = record[0], record[1], record[2]
            if self.has_category:
                category = record[3]
                if pd.isnull(attribute_data.get(uid)) or pd.isnull(category):
                    continue
            else:
                if pd.isnull(attribute_data.get(uid)):
                    continue
            grouped_indices[attribute_data.get(uid)].append(idx)

        #Flatten the list of indices and sort by attribute value
        sorted_indices = sorted((idx for indices in grouped_indices.values() for idx in indices),
                            key=lambda idx: attribute_data.get(data1[idx][1],0))
        total_size = len(sorted_indices)
        bin_size = total_size // self.num_bins
        leftover = total_size % self.num_bins
        # Split indices into bins
        bins = []
        start_index = 0
        for bin_num in range(self.num_bins):
            if bin_num < leftover:
                end_index = start_index + bin_size + 1
            else:
                end_index = start_index + bin_size
            bins.append(sorted_indices[start_index:end_index])
            start_index = end_index
    # Sort the data points based on the attribute value
        train_idx, val_idx, test_idx = [], [], []
        for bin_indices in bins:
            np.random.shuffle(bin_indices)
            cut_point = int(len(bin_indices) * self.train_size)
            train_idx.extend(bin_indices[:cut_point])
            test_idx.extend(bin_indices[cut_point:])

        # Extract data for training and testing
        train_data = [data1[i] for i in train_idx]
        test_data = [data1[i] for i in test_idx]
        val_data = None  # If validation data is needed, add logic here
        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
        self.print_statistics(train_data, test_data,val_data)
        print(f"Final dataset sizes - Train: {len(train_data)}, Test: {len(test_data)}, Val: None")
    
    def print_statistics(self, train_data, test_data, val_data):
        num_train_users = len(set(row[0] for row in train_data)) if train_data else 0
        num_train_items = len(set(row[1] for row in train_data)) if train_data else 0
        num_train_ratings = len(train_data)
        max_train_rating = max((row[2] for row in train_data), default=0)
        min_train_rating = min((row[2] for row in train_data), default=0)
        mean_train_rating = np.mean([row[2] for row in train_data]) if train_data else 0

        num_test_users = len(set(row[0] for row in test_data)) if test_data else 0
        num_test_items = len(set(row[1] for row in test_data)) if test_data else 0
        num_test_ratings = len(test_data)
        max_test_rating = max((row[2] for row in test_data), default=0)
        min_test_rating = min((row[2] for row in test_data), default=0)
        mean_test_rating = np.mean([row[2] for row in test_data]) if test_data else 0

        print("Training data:")
        print(f"Number of users = {num_train_users}")
        print(f"Number of items = {num_train_items}")
        print(f"Number of ratings = {num_train_ratings}")
        print(f"Max rating = {max_train_rating}")
        print(f"Min rating = {min_train_rating}")
        print(f"Global mean = {mean_train_rating:.4f}")

        print("---")
        print("Test data:")
        print(f"Number of users = {num_test_users}")
        print(f"Number of items = {num_test_items}")
        print(f"Number of ratings = {num_test_ratings}")
        print(f"Max rating = {max_test_rating}")
        print(f"Min rating = {min_test_rating}")
        print(f"Global mean = {mean_test_rating:.4f}")

        if val_data is not None:
            num_val_users = len(set([row[0] for row in val_data]))
            num_val_items = len(set([row[1] for row in val_data]))
            num_val_ratings = len(val_data)
            max_val_rating = max([row[2] for row in val_data]) if val_data else None
            min_val_rating = min([row[2] for row in val_data]) if val_data else None
            mean_val_rating = np.mean([row[2] for row in val_data]) if val_data else None
            print("---")
            print("Validation data:")
            print(f"Number of users = {num_val_users}")
            print(f"Number of items = {num_val_items}")
            print(f"Number of ratings = {num_val_ratings}")
            print(f"Max rating = {max_val_rating}")
            print(f"Min rating = {min_val_rating}")
            print(f"Global mean = {mean_val_rating:.4f}" if mean_val_rating else "Global mean = None")
        else:
            print("---")
            print("No validation data.")

        if self.output_test_csv:
            output_dir = os.path.dirname(self.output_test_csv)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            test_df = pd.DataFrame(test_data, columns=['user', 'item', 'rating',  'category'])
            test_df.to_csv(self.output_test_csv, index=False)
            print(f"Testing data saved to {self.output_test_csv}")


class StratifiedDiversityPart(BaseMethod):
    def __init__(
        self,
        data1,
        data2=None,
        data3=None,
        data4=None,
        group_by="sentiment",
        fmt="UIR",
        train_size=0.8,
        test_size=0.2,
        val_size=0.0,
        seed=None,
        rating_threshold=0.5,
        exclude_unknowns=True,
        verbose=False,
        output_test_csv=None,
        num_bins=10,  # number of levels for diversity scores
        top_percentage=50,  # percentage of data to keep
        order="desc",  # "desc" for descending, "asc" for ascending
        **kwargs
    ):
        super().__init__(
            data=data1,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            output_test_csv=output_test_csv,
            **kwargs
        )
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.group_by = group_by
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.output_test_csv = output_test_csv
        self.num_bins = num_bins
        self.top_percentage = top_percentage
        self.order = order.lower()  # Ensure the order is in lowercase

        if group_by not in ["category", "complexity", "sentiment"]:
            raise ValueError("group_by option must be 'category', 'complexity', or 'sentiment'")

        if self.order not in ["asc", "desc"]:
            raise ValueError("order option must be either 'asc' for ascending or 'desc' for descending")

        # Check if data1 has a category column
        self.has_category = len(data1[0]) == 4 if data1 else False

        self.split()

    def split(self):
        print(f"Initial data sizes - data1: {len(self.data1)}, data2: {len(self.data2 if self.data2 else [])}, data3: {len(self.data3 if self.data3 else [])}")
        data1 = self.data1
        data2 = self.data2
        data3 = self.data3
        data4 = self.data4
    
        # Select the attribute data to group by
        if self.group_by == "sentiment":
            attribute_data = data2
        elif self.group_by == "complexity":
            attribute_data = data3
        elif self.group_by == "category":
            attribute_data = data4
    
        # Create a dictionary to store the indices of the data points
        grouped_indices = defaultdict(list)
        for idx, record in enumerate(data1):
            if self.has_category:
                uid, iid, rating, category = record
                if pd.isnull(attribute_data.get(uid)) or pd.isnull(category):
                    continue
            else:
                uid, iid, rating = record[:3]
                category = None
                if pd.isnull(attribute_data.get(uid)):
                    continue

            grouped_indices[attribute_data.get(uid)].append(idx)

        # Determine sort order based on 'order' parameter
        reverse_order = self.order == "desc"
        
        # Flatten the list of indices and sort by attribute value
        sorted_indices = sorted(
            (idx for indices in grouped_indices.values() for idx in indices),
            key=lambda idx: attribute_data.get(data1[idx][0], 0),
            reverse=reverse_order
            )
        print("Top 10 sorted data1 records after sorting and filtering:")
        for i in range(min(10, len(sorted_indices))):
            print(data1[sorted_indices[i]])

        # Calculate the number of data points to keep
        num_to_keep = int(len(sorted_indices) * self.top_percentage / 100)
        top_indices = sorted_indices[:num_to_keep]

        # Divide the data points into bins
        bins = []
        bin_size = len(top_indices) // self.num_bins
        leftover = len(top_indices) % self.num_bins
        start_index = 0
        for bin_num in range(self.num_bins):
            if bin_num < leftover:
                end_index = start_index + bin_size + 1
            else:
                end_index = start_index + bin_size
            bins.append(top_indices[start_index:end_index])
            start_index = end_index

        # Sort the data points based on the attribute value
        train_idx, test_idx = [], []
        for bin_indices in bins:
            np.random.shuffle(bin_indices)
            cut_point = int(len(bin_indices) * self.train_size)
            train_idx.extend(bin_indices[:cut_point])
            test_idx.extend(bin_indices[cut_point:])

        # Extract data for training and testing
        train_data = [data1[i] for i in train_idx]
        test_data = [data1[i] for i in test_idx]
        val_data = None  
        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
        self.print_statistics(train_data, test_data, val_data)
        print(f"Final dataset sizes - Train: {len(train_data)}, Test: {len(test_data)}, Val: None")

    def print_statistics(self, train_data, test_data, val_data):
        num_train_users = len(set(row[0] for row in train_data)) if train_data else 0
        num_train_items = len(set(row[1] for row in train_data)) if train_data else 0
        num_train_ratings = len(train_data)
        max_train_rating = max((row[2] for row in train_data), default=0)
        min_train_rating = min((row[2] for row in train_data), default=0)
        mean_train_rating = np.mean([row[2] for row in train_data]) if train_data else 0

        num_test_users = len(set(row[0] for row in test_data)) if test_data else 0
        num_test_items = len(set(row[1] for row in test_data)) if test_data else 0
        num_test_ratings = len(test_data)
        max_test_rating = max((row[2] for row in test_data), default=0)
        min_test_rating = min((row[2] for row in test_data), default=0)
        mean_test_rating = np.mean([row[2] for row in test_data]) if test_data else 0

        print("Training data:")
        print(f"Number of users = {num_train_users}")
        print(f"Number of items = {num_train_items}")
        print(f"Number of ratings = {num_train_ratings}")
        print(f"Max rating = {max_train_rating}")
        print(f"Min rating = {min_train_rating}")
        print(f"Global mean = {mean_train_rating:.4f}")

        print("---")
        print("Test data:")
        print(f"Number of users = {num_test_users}")
        print(f"Number of items = {num_test_items}")
        print(f"Number of ratings = {num_test_ratings}")
        print(f"Max rating = {max_test_rating}")
        print(f"Min rating = {min_test_rating}")
        print(f"Global mean = {mean_test_rating:.4f}")

        if val_data is not None:
            num_val_users = len(set([row[0] for row in val_data]))
            num_val_items = len(set([row[1] for row in val_data]))
            num_val_ratings = len(val_data)
            max_val_rating = max([row[2] for row in val_data]) if val_data else None
            min_val_rating = min([row[2] for row in val_data]) if val_data else None
            mean_val_rating = np.mean([row[2] for row in val_data]) if val_data else None
            print("---")
            print("Validation data:")
            print(f"Number of users = {num_val_users}")
            print(f"Number of items = {num_val_items}")
            print(f"Number of ratings = {num_val_ratings}")
            print(f"Max rating = {max_val_rating}")
            print(f"Min rating = {min_val_rating}")
            print(f"Global mean = {mean_val_rating:.4f}" if mean_val_rating else "Global mean = None")
        else:
            print("---")
            print("No validation data.")

        if self.output_test_csv:
            output_dir = os.path.dirname(self.output_test_csv)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            columns = ['user', 'item', 'rating'] + (['category'] if self.has_category else [])
            test_df = pd.DataFrame(test_data, columns=columns)
            test_df.to_csv(self.output_test_csv, index=False)
            print(f"Testing data saved to {self.output_test_csv}")

class StratifiedByClusters(BaseMethod):
    def __init__(
        self,
        data1,
        data2,
        train_size=0.8,
        test_size=0.2,
        val_size=0.0,
        group_by="clusters",
        fmt="UIRT",
        seed=None,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
        output_test_csv=None,
        **kwargs
    ):
        """
        Initialize the StratifiedByClusters class.

        Parameters
        ----------
        data1 : list
            The primary dataset containing user-item-rating tuples.
        data2 : pd.DataFrame
            The dataset containing User_ID and clusters.
        train_size : float
            Proportion of data to use for training.
        test_size : float
            Proportion of data to use for testing.
        val_size : float
            Proportion of data to use for validation (if needed).
        seed : int
            Random seed for shuffling the data.
        """
        super().__init__(
            data=data1,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            output_test_csv=None,
            **kwargs
        )
        self.data1 = data1
        self.data2 = data2
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.group_by = group_by
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.output_test_csv = output_test_csv
        if group_by not in ["clusters"]:
            raise ValueError("group_by option must be 'clusters'")
        self.has_category = len(data1[0]) == 4 if data1 else False

        self.split()

    def split(self):
        """
        Perform stratified split based on cluster information.
        """
        print(f"Initial data size: {len(self.data1)}")
        
        data1 = self.data1
        data2 = self.data2

    
    # Choose the attribute data to group by
        if self.group_by == "clusters":
            attribute_data = data2

        grouped_indices = defaultdict(list)

        # Create a dictionary to store the indices of the data points
        for idx, record in enumerate(data1):
            if self.has_category:
                uid, iid, rating, category = record
                if pd.isnull(data2.get(uid)) or pd.isnull(category):
                    continue
            else:
                uid, iid, rating = record[:3]
                category = None
                if pd.isnull(data2.get(uid)):
                    continue

            cluster = data2.get(uid)
            if cluster is not None:
                grouped_indices[cluster].append(idx)

        train_idx, val_idx, test_idx = [], [], []
        
        rng = np.random.default_rng(self.seed)
        
        for clusters, indices in grouped_indices.items():
            n_items = len(indices)  

            # For clusters with less than 10 items, add all items to the training set
            if n_items < 10:
                train_idx.extend(indices)
            else:
                n_train = max(1, int(n_items * self.train_size)) 
                n_test = max(1, int(n_items * self.test_size))   
                n_val = n_items - n_train - n_test 
                if n_train == 0 and n_test == 0:
                    train_idx.extend(indices)
                else:
                # Shuffle the indices
                    shuffled_indices = rng.permutation(indices)
                    train_idx.extend(shuffled_indices[:n_train])
                    val_idx.extend(shuffled_indices[n_train:n_train + n_val])
                    test_idx.extend(shuffled_indices[n_train + n_val:])
        # Extract data for training and testing
        train_data = [self.data1[i] for i in train_idx]
        val_data = [self.data1[i] for i in val_idx] if val_idx else None
        test_data = [self.data1[i] for i in test_idx]
        
        self.build(train_data=train_data, test_data=test_data, val_data=val_data)

        # print statistics
        self.print_statistics(train_data, test_data, val_data)
        print(f"Final dataset sizes - Train: {len(train_data)}, Test: {len(test_data)}, Val: None")

    def print_statistics(self, train_data, test_data, val_data):
        """
        Print statistics for the train, test, and validation datasets.
        """
        num_train_users = len(set(row[0] for row in train_data)) if train_data else 0
        num_train_items = len(set(row[1] for row in train_data)) if train_data else 0
        num_train_ratings = len(train_data)

        num_test_users = len(set(row[0] for row in test_data)) if test_data else 0
        num_test_items = len(set(row[1] for row in test_data)) if test_data else 0
        num_test_ratings = len(test_data)

        print("Training data:")
        print(f"Number of users = {num_train_users}")
        print(f"Number of items = {num_train_items}")
        print(f"Number of ratings = {num_train_ratings}")

        print("---")
        print("Test data:")
        print(f"Number of users = {num_test_users}")
        print(f"Number of items = {num_test_items}")
        print(f"Number of ratings = {num_test_ratings}")

        if val_data is not None:
            num_val_users = len(set(row[0] for row in val_data))
            num_val_items = len(set(row[1] for row in val_data))
            num_val_ratings = len(val_data)
            print("---")
            print("Validation data:")
            print(f"Number of users = {num_val_users}")
            print(f"Number of items = {num_val_items}")
            print(f"Number of ratings = {num_val_ratings}")
        else:
            print("---")
            print("No validation data.")
