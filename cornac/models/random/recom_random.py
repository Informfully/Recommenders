# Copyright 2018 The Cornac Authors. All Rights Reserved.
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

import numpy as np

from ..recommender import Recommender
from ...exception import ScoreException
import random

class RandomModel(Recommender):
    """Random. Item are recommended randomly.

    Change: can perform ranking within an item pool.

    Parameters
    ----------
    name: string, default: 'RandomModel'
        The name of the recommender model.
    """

    def __init__(self, name="RandomModel", userHistory = None,     **kwargs):
        
        super().__init__(name=name,   **kwargs)
        self.userHistory = userHistory



    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        return self



    def score(self, user_idx, item_idx=None, **kwargs):
        if item_idx is None:
            prediction_scores = np.random.rand(self.num_items)
        
        else: 
            # Generate random prediction scores
            prediction_scores = np.random.rand(len(item_idx)) 
        return prediction_scores
    
