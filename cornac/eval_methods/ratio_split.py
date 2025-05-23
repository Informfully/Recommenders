# Copyright 2018 The Cornac Authors. All Rights Reserved.
############################################################################
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from math import ceil

from .base_method import BaseMethod
from ..utils.common import safe_indexing
from .static_rerank_evaluator import StaticReRankEval
from .dynamic_rerank_evaluator import DynamicReRankEval
from ..experiment.result import Result


class RatioSplit(BaseMethod):
    """Splitting data into training, validation, and test sets based on provided sizes.
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

    """

    def __init__(
        self,
        data,
        test_size=0.2,
        val_size=0.0,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            data=data,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs,
        )

        self.train_size, self.val_size, self.test_size = self.validate_size(
            val_size=val_size,
            test_size=test_size,
            data_size=kwargs.get("data_size", len(data)),
        )
        self._split()

    @staticmethod
    def validate_size(val_size, test_size, data_size):
        if val_size is None:
            val_size = 0.0
        elif val_size < 0:
            raise ValueError("val_size={} should be greater than zero".format(val_size))
        elif val_size >= data_size:
            raise ValueError(
                f"val_size={val_size} should be smaller than data_size={data_size}"
            )

        if test_size is None:
            test_size = 0.0
        elif test_size < 0:
            raise ValueError(f"test_size={test_size} should be greater than zero")
        elif test_size >= data_size:
            raise ValueError(
                f"test_size={test_size} should be smaller than data_size={data_size}"
            )

        if val_size < 1:
            val_size = ceil(val_size * data_size)
        if test_size < 1:
            test_size = ceil(test_size * data_size)

        val_test_size = val_size + test_size
        if val_test_size >= data_size:
            raise ValueError(
                f"val_size + test_size ({val_test_size}) should be smaller than data_size={data_size}"
            )

        train_size = data_size - (val_size + test_size)

        return int(train_size), int(val_size), int(test_size)

    def _split(self):
        data_idx = self.rng.permutation(len(self.data))
        train_idx = data_idx[: self.train_size]
        test_idx = data_idx[-self.test_size :]
        val_idx = data_idx[self.train_size : -self.test_size]

        train_data = safe_indexing(self.data, train_idx)
        test_data = safe_indexing(self.data, test_idx)
        val_data = safe_indexing(self.data, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data,
                   test_data=test_data, val_data=val_data)







      


    