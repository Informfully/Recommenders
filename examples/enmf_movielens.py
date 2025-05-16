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
"""Example for Denoising Autoencoder, using the Movielens dataset"""

import cornac
from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE,RMSE,MSE,FMeasure,Precision,Recall,NDCG,NCRR,MRR,AUC,MAP
from cornac.models import ENMF
# Load user-item feedback
data = cornac.datasets.movielens.load_feedback(variant="100K")

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

# Instantiate the ENMF model
enmf = ENMF(
  num_epochs = 10
)

# Instantiate evaluation measures
metrics = [MAE(), RMSE(), MSE(),FMeasure(k=100),Precision(k=100),
           Recall(k=100), NDCG(k=100), NCRR(k=100),
           MRR(),AUC(), MAP()]

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[enmf],
    metrics=metrics,
    user_based=True,
).run()


