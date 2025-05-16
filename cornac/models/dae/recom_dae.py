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

class DAE(Recommender):
    """Denoising Autoencoder.


    Parameters
    ----------
    qk_dims: list, default: 200
        The dimension of encoder layer for DAE
        For example, qk_dims = [200], the DAE encoder structure will be [n_items, 200]

    pk_dims: list, default: 200
        The dimension of decoder layer for DAE
        For example, pk_dims = [200], the DAE decoder structure will be [200, n_items]

    n_epochs: int, optional, default: 100
        The number of epochs for SGD.

    batch_size: int, optional, default: 100
        The batch size.

    learning_rate: float, optional, default: 0.001
        The learning rate for Adam.

    weight_decay: float, optional, default: 0.0
        A regularization parameter.

    dropout_p:  float, optional, default: 0.5
        A technique for preventing the network from overfitting.

    name: string, optional, default: 'DAE'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    use_gpu: boolean, optional, default: False
        If True and your system supports CUDA then training is performed on GPUs.

    References
    ----------
    * Liang, Dawen, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. "Variational autoencoders for collaborative filtering." \
    In Proceedings of the 2018 World Wide Web Conference on World Wide Web, pp. 689-698.
    """

    def __init__(
            self,
            name="DAE",
            qk_dims = [200],
            pk_dims = [200],
            n_epochs=100,
            batch_size=100,
            learning_rate=0.001,
            weight_decay=0.0,
            dropout_p=0.5,
            trainable=True,
            verbose=False,
            seed=None,
            use_gpu=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.qk_dims = qk_dims
        self.pk_dims = pk_dims
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.seed = seed
        self.use_gpu = use_gpu

    def fit(self, train_set, val_set=None):
        """Fit the model to observatinos.

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

        import torch
        from .dae import DAE, learn

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "dae"):
                data_dim = train_set.matrix.shape[1]
                self.q_dims = [data_dim] + self.qk_dims
                self.p_dims = self.pk_dims + [data_dim]
                self.dae = DAE(
                    self.weight_decay,
                    self.dropout_p,
                    self.q_dims,
                    self.p_dims,
                ).to(self.device)

            learn(
                self.dae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                verbose=self.verbose,
                device=self.device,
            )

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self

    def score(self, user_idx, item_idx=None, **kwargs):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        import torch
        # from .dae import Encoder, Decoder

        # encoder = Encoder(None, dropout_p=self.dropout_p, q_dims=self.q_dims)
        # decoder = Decoder(None, p_dims=self.p_dims)
        if not self.knows_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )
        if item_idx is not None and not self.knows_item(item_idx):
            raise ScoreException(
                "Can't make score prediction for (item_id=%d)" % item_idx
            )

        if item_idx is None:
        
            x_u = self.train_set.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))
            z_u = self.dae.forward(
                torch.tensor(x_u.toarray(), dtype=torch.float32, device=self.device)
            )
            known_item_scores = z_u.data.cpu().numpy().flatten()

            return known_item_scores
        else:
            # Ensure item_idx can be either a single int, list, or numpy array
            if isinstance(item_idx, (int, np.integer)):
                item_idx = [item_idx]  # Convert single int to list for consistency
            elif not isinstance(item_idx, (list, np.ndarray)):
                raise ValueError("item_idx should be an int, list, or numpy array.")

            # Verify that none of the items are unknown
            if any(self.knows_item(idx) for idx in item_idx):
                raise ScoreException(
                    "Can't make score prediction for some unknown item(s)"
                )
             # Create input vector for the user
            x_u = self.train_set.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))

            # Forward pass through the model to get predictions
            z_u = self.dae.forward(
                torch.tensor(x_u.toarray(), dtype=torch.float32, device=self.device)
            )

            # Extract the predictions only for the given item indices
            user_pred = z_u.data.cpu().numpy().flatten()[item_idx]

            return user_pred if len(user_pred) > 1 else user_pred[0]  # Return scalar if single item
            



