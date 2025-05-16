# License
# ============================================================================
from tqdm.auto import trange
# import tensorflow as tf
from ..recommender import Recommender
# from ...utils import get_rng
import tensorflow.compat.v1 as tf
from ...exception import ScoreException
import numpy as np
import pickle
import os
from datetime import datetime


class ENMF(Recommender):
    """class of ENMF.

    Parameters
    ----------
    embedding_size: int, optional, default: 64
        Embedding size of ENMF model.

    lambda_bilinear: list, optional, default: [0.0, 0.0].
        Regularization for user and item embeddings. 

    num_epochs: int, optional, default: 100
        Number of epochs.

    batch_size: int, optional, default: 256
        Batch size.

    neg_weight: 0-1, optional, default: 0.5
        Negative weight.

    lr: float, optional, default: 0.05
        Learning rate.

    dropout_p:  float, optional, default: 0.7
    Dropout keep probability. The probability of retaining a neuron or unit during training.

    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments: 

         - `min_delta`: the minimum increase in monitored value on validation set to be considered as improvement, \
           i.e. an increment of less than min_delta will count as no improvement.
         - `patience`: number of epochs with no improvement after which training should be stopped.

    name: string, optional, default: 'ENMF'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    References
    ----------
    * Chen, C., Zhang, M., Zhang, Y., Liu, Y., & Ma, S. (2020). Efficient neural matrix factorization without sampling for recommendation. 
    ACM Transactions on Information Systems (TOIS), 38(2), 1-28.
    """

    def __init__(
        self,
        name="ENMF",
        embedding_size=64,
        num_epochs=100,
        batch_size=256,
        neg_weight=0.5,
        lambda_bilinear=[0.0, 0.0],
        lr=0.05,
        dropout_p=0.7,
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=2019,
    ):
        Recommender.__init__(
            self, name=name, trainable=trainable, verbose=verbose)
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.neg_weight = neg_weight
        self.lambda_bilinear = lambda_bilinear
        self.lr = lr
        self.early_stopping = early_stopping
        self.max_item_pu = 0
        self.dropout_p = dropout_p
        self.seed = seed
        self.ignored_attrs.extend(
            [
                "graph",
                "user_id",
                "item_id",
                "labels",
                "interaction",
                "prediction",
                "loss",
                "train_op",
                "initializer",
                "saver",
                "sess",
            ]
        )

    def _create_placeholders(self):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_ur = tf.placeholder(
            tf.int32, [None, self.max_item_pu], name="input_ur")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    def _create_variables(self):
        self.uidW = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.iidW = tf.Variable(tf.truncated_normal(shape=[self.num_items + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")

        self.H_i = tf.Variable(tf.constant(
            0.01, shape=[self.embedding_size, 1]), name="hi")

    def _create_inference(self):
        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.embedding_size])

        self.uid = tf.nn.dropout(self.uid, rate=1-self.dropout_keep_prob)
        self.pos_item = tf.nn.embedding_lookup(self.iidW, self.input_ur)
        self.pos_num_r = tf.cast(tf.not_equal(
            self.input_ur, self.num_items), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
        self.pos_r = tf.einsum('ac,abc->abc', self.uid, self.pos_item)
        self.pos_r = tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
        self.pos_r = tf.reshape(self.pos_r, [-1, self.max_item_pu])

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
        def interacted_items(csr_row):
            return [
                item_idx
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                if rating > 0
            ]
        Recommender.fit(self, train_set, val_set)
        gt_mat = train_set.csr_matrix
        self.train_set = train_set
        self.val_set = val_set
        self.train_set_dict = {}
        max_item_pu = 0
        train_user_indices = set(train_set.uir_tuple[0])
        for user_idx in train_user_indices:
            train_pos_items = interacted_items(gt_mat.getrow(user_idx))
            self.train_set_dict[user_idx] = train_pos_items
        for i in self.train_set_dict:
            if len(self.train_set_dict[i]) > max_item_pu:
                max_item_pu = len(self.train_set_dict[i])
        self.max_item_pu = max_item_pu
        for i in self.train_set_dict:
            while len(self.train_set_dict[i]) < max_item_pu:
                self.train_set_dict[i].append(train_set.num_items)
        if self.trainable:
            if not hasattr(self, "graph"):
                self.num_users = self.train_set.num_users
                self.num_items = self.train_set.num_items
                self._build_graph()

            self._fit_tf()
        return self

    def _pre(self):
        dot = tf.einsum('ac,bc->abc', self.uid, self.iidW)
        pre = tf.einsum('ajk,kl->ajl', dot, self.H_i)
        pre = tf.reshape(pre, [-1, self.num_items + 1])
        return pre

    def _create_loss(self):
        self.loss1 = self.neg_weight * tf.reduce_sum(
            tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc', self.iidW, self.iidW), 0)
                          * tf.reduce_sum(tf.einsum('ab,ac->abc', self.uid, self.uid), 0)
                          * tf.matmul(self.H_i, self.H_i, transpose_b=True), 0), 0)
        self.loss1 += tf.reduce_sum((1.0 - self.neg_weight)
                                    * tf.square(self.pos_r) - 2.0 * self.pos_r)
        self.l2_loss0 = tf.nn.l2_loss(self.uidW)
        self.l2_loss1 = tf.nn.l2_loss(self.iidW)
        self.loss = self.loss1 \
            + self.lambda_bilinear[0] * self.l2_loss0 \
            + self.lambda_bilinear[1] * self.l2_loss1

        self.reg_loss = self.lambda_bilinear[0] * self.l2_loss0 \
            + self.lambda_bilinear[1] * self.l2_loss1

    def _build_graph(self):
        with tf.Graph().as_default():
            self._sess_init()
            with self.sess.as_default():
                self._create_placeholders()
                self._create_variables()
                self._create_inference()
                self._create_loss()
                self.prediction = self._pre()
                self.train_op = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(
                    self.loss)
                self.saver = tf.train.Saver()
                self.initializer = tf.global_variables_initializer()

    def _sess_init(self):
        import tensorflow.compat.v1 as tf

        tf.set_random_seed(self.seed)
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_conf)

    def _step_update(self, batch_users, batch_items):
        if batch_items.shape[1] != self.max_item_pu:
            print(f"Warning: batch_items shape {batch_items.shape} doesn't match max_item_pu {self.max_item_pu}")

        feed_dict = {
            self.input_u: batch_users,
            self.input_ur: batch_items,
            self.dropout_keep_prob: self.dropout_p,
        }
        _, loss, loss1, loss2 = self.sess.run(
            [self.train_op, self.loss, self.loss1, self.reg_loss],
            feed_dict)
        return loss, loss1, loss2

    def get_train_instances1(self, train_set_dict):
        user_train, item_train = [], []
        for i in train_set_dict.keys():
            user_train.append(i)
            item_train.append(train_set_dict[i])

        user_train = np.array(user_train)
        item_train = np.array(item_train)
        user_train = user_train[:, np.newaxis]
        return user_train, item_train

    def _fit_tf(self):
        self.sess.run(self.initializer)
        user_train1, item_train1 = self.get_train_instances1(
            self.train_set_dict)
        for epoch in range(self.num_epochs):
            print(epoch)

            shuffle_indices = np.random.permutation(
                np.arange(len(user_train1)))
            user_train1 = user_train1[shuffle_indices]
            item_train1 = item_train1[shuffle_indices]
            ll = int(len(user_train1) / self.batch_size)
            loss = [0.0, 0.0, 0.0]
            for batch_num in range(ll):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) *
                                self.batch_size, len(user_train1))

                u_batch = user_train1[start_index:end_index]
                i_batch = item_train1[start_index:end_index]
                loss1, loss2, loss3 = self._step_update(u_batch, i_batch)
                loss[0] += loss1
                loss[1] += loss2
                loss[2] += loss3
            if self.early_stopping is not None and self.early_stop(
                **self.early_stopping
            ):
                break

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
        if save_dir is None:
            return
        tf.compat.v1.disable_v2_behavior()

        model_dir = os.path.join(save_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        model_file = os.path.join(model_dir, "{}.ckpt".format(timestamp))
        self.saver.save(
            self.sess, model_file)

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default,
            the model parameters are assumed to be fixed after being loaded.

        Future development.
        -------
        self : object
        """
        return

    def monitor_value(self):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if self.val_set is None:
            return None

        from ...metrics import NDCG
        from ...eval_methods import ranking_eval

        ndcg_100 = ranking_eval(
            model=self,
            metrics=[NDCG(k=100)],
            train_set=self.train_set,
            test_set=self.val_set,
        )[0][0]

        return ndcg_100

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
        if item_idx is None:
            if self.is_unknown_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            user_te = np.array([user_idx])
            user_te2 = user_te[:, np.newaxis]
            known_item_scores = self.sess.run(
                self.prediction,
                feed_dict={
                    self.input_u: user_te2,
                    self.dropout_keep_prob: 1.0,
                },
            )
            return known_item_scores.ravel()[:-1]
        else:
            if self.is_unknown_user(user_idx) or self.is_unknown_item(item_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            user_te = np.array([user_idx])
            user_te2 = user_te[:, np.newaxis]
            user_pred = self.sess.run(
                self.prediction,
                feed_dict={self.input_u: user_te2,
                           self.dropout_keep_prob: 1.0, },
            )
            user_result = user_pred.ravel()[:-1]
            return user_result[item_idx]
