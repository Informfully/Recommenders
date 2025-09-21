# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class AttLayer2(layers.Layer):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=200, seed=0, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            dim (int): attention hidden dim
        """
        self.dim = dim
        self.seed = seed
        super(AttLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initialization for variables in AttLayer2
        There are there variables in AttLayer2, i.e. W, b and q.

        Args:
            input_shape (object): shape of input tensor.
        """
        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(dim,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(dim, 1),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super(AttLayer2, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        """Core implementation of soft attention.

        Args:
            inputs (object): input tensor.

        Returns:
            object: weighted sum of input tensors.
        """
        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)
        attention = K.squeeze(attention, axis=2)

        if mask is None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * tf.cast(mask, dtype=tf.float32)

        attention_weight = attention / (
            K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
        )

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, input, input_mask=None):
        """Compte output mask value.

        Args:
            input (object): input tensor.
            input_mask: input mask

        Returns:
            object: output mask.
        """
        return None

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Args:
            input_shape (tuple): shape of input tensor.

        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]


class SelfAttention(layers.Layer):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimension of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, **kwargs):
        """Initialization steps for SelfAttention.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimension of each head.
            mask_right (boolean): Whether to mask right words.
        """
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Returns:
            tuple: output shape tuple.
        """
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e. WQ, WK ans WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (object): shape of input tensor.
        """
        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super(SelfAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.

        Returns:
            object: tensors after masking.
        """
        if seq_len is None:
            return inputs
        else:
            mask = tf.one_hot(indices=seq_len[:, 0], depth=tf.shape(inputs)[1])
            mask = 1 - tf.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = tf.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. query, key and value.

        Returns:
            object: ouput tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = tf.reshape(
            Q_seq, shape=(-1, tf.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = tf.transpose(Q_seq, perm=[0, 2, 1, 3])

        K_seq = K.dot(K_seq, self.WK)
        K_seq = tf.reshape(
            K_seq, shape=(-1, tf.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = tf.transpose(K_seq, perm=[0, 2, 1, 3])

        V_seq = K.dot(V_seq, self.WV)
        V_seq = tf.reshape(
            V_seq, shape=(-1, tf.shape(V_seq)[1], self.multiheads, self.head_dim)
        )
        V_seq = tf.transpose(V_seq, perm=[0, 2, 1, 3])

        A = tf.einsum("abij,abkj->abik", Q_seq, K_seq) / tf.sqrt(
            tf.cast(self.head_dim, dtype=tf.float32)
        )
        A = tf.transpose(A, perm=[0, 3, 2, 1])

        A = self.Mask(A, V_len, "add")
        A = tf.transpose(A, perm=[0, 3, 2, 1])

        if self.mask_right:
            ones = tf.ones_like(A[:1, :1])
            lower_triangular = tf.linalg.band_part(ones, num_lower=-1, num_upper=0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = tf.nn.softmax(A)

        O_seq = tf.einsum("abij,abjk->abik", A, V_seq)
        O_seq = tf.transpose(O_seq, perm=[0, 2, 1, 3])

        O_seq = tf.reshape(O_seq, shape=(-1, tf.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq

    def get_config(self):
        """add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.
        """
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
                "seed": self.seed,
            }
        )
        return config


def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implement.

    Attributes:
        dim1 (int): first dimention of value shape.
        dim2 (int): second dimention of value shape.
        dim3 (int): shape of query

    Returns:
        object: weighted summary of inputs value.
    """
    vecs_input = tf.keras.Input(shape=(dim1, dim2), dtype=tf.float32)
    query_input = tf.keras.Input(shape=(dim3,), dtype=tf.float32)

    user_vecs = layers.Dropout(0.2)(vecs_input)
    user_att = layers.Dense(
        dim3,
        activation="tanh",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros(),
    )(user_vecs)
    user_att2 = layers.Dot(axes=-1)([query_input, user_att])
    user_att2 = layers.Activation("softmax")(user_att2)
    user_vec = layers.Dot((1, 1))([user_vecs, user_att2])

    model = tf.keras.Model([vecs_input, query_input], user_vec)
    return model


class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """Call method for ComputeMasking.

        Args:
            inputs (object): input tensor.

        Returns:
            bool tensor: True for values not equal to zero.
        """
        mask = tf.not_equal(inputs, 0)
        return tf.cast(mask, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at specific positions to zero.

    Args:
        inputs (list): value tensor and mask tensor.

    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Call method for OverwriteMasking.

        Args:
            inputs (list): value tensor and mask tensor.

        Returns:
            object: tensor after setting values to zero.
        """
        return inputs[0] * tf.expand_dims(inputs[1], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]