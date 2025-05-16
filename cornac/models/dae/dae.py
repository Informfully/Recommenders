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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm.auto import trange

from ...utils import estimate_batches

torch.set_default_dtype(torch.float32)

EPS = 1e-10
n_items = 1e-10
n_dim = 200

class Encoder(nn.Module):
    def __init__(self, options, dropout_p=0.5, q_dims=[n_items, n_dim]):
        super(Encoder, self).__init__()
        self.options = options
        self.q_dims = q_dims

        self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        self.linear_1 = nn.Linear(self.q_dims[0], self.q_dims[1], bias=True)
        self.tanh = nn.Tanh()

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, options, p_dims=[n_dim, n_items]):
        super(Decoder, self).__init__()
        self.options = options
        self.p_dims = p_dims

        self.linear_1 = nn.Linear(self.p_dims[0], self.p_dims[1], bias=True)
        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.linear_1(x)
        return x

class DAE(nn.Module):
    def __init__(self, weight_decay=0.0, dropout_p=0.5, q_dims=[n_items, n_dim], p_dims=[n_dim, n_items]):
        super(DAE, self).__init__()
        self.weight_decay = weight_decay

        self.encoder = Encoder(None, dropout_p=dropout_p, q_dims=q_dims)
        self.decoder = Decoder(None, p_dims=p_dims)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.encoder.forward(x)
        logits = self.decoder.forward(x)
        return logits

    def get_l2_reg(self):
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        if self.weight_decay > 0:
            for k, m in self.state_dict().items():
                if k.endswith('.weight'):
                    l2_reg = l2_reg + torch.norm(m, p=2) ** 2
            l2_reg = self.weight_decay * l2_reg
        return l2_reg[0]

    def loss(self, x, x_):
        log_softmax_var = F.log_softmax(x_, dim=1)
        neg_ll = - torch.mean(torch.sum(log_softmax_var * x, dim=1))
        l2_reg = self.get_l2_reg()

        return neg_ll + l2_reg


def learn(
    dae,
    train_set,
    n_epochs,
    batch_size,
    learn_rate,
    verbose,
    device=torch.device("cpu"),
):
    optimizer = torch.optim.Adam(params=dae.parameters(), lr=learn_rate)
    num_steps = estimate_batches(train_set.num_users, batch_size)

    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        sum_loss = 0.0
        count = 0
        for batch_id, u_ids in enumerate(
                train_set.user_iter(batch_size, shuffle=False)
        ):
            u_batch = train_set.matrix[u_ids, :]
            u_batch.data = np.ones(len(u_batch.data))  # Binarize data
            u_batch = u_batch.toarray()
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=device)

            # Reconstructed batch
            u_batch_ = dae(u_batch)

            loss = dae.loss(u_batch, u_batch_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))

    return dae
