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

# Instantiate the DAE model
dae = cornac.models.DAE(
    qk_dims=[200],
    pk_dims=[200],
    n_epochs=100,
    batch_size=100,
    learning_rate=0.001,
    weight_decay=0.0,
    dropout_p=0.5,
    seed=123,
    use_gpu=True,
    verbose=True,
)

# Instantiate evaluation measures
metrics = [MAE(), RMSE(), MSE(),FMeasure(k=200),Precision(k=200),
           Recall(k=200), NDCG(k=200), NCRR(k=200),
           MRR(),AUC(), MAP()]

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[dae],
    metrics=metrics,
    user_based=True,
).run()


