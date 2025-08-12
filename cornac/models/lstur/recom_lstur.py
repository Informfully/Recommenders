#############################################################################################################
# Model name:      LSTUR model(Neural News Recommendation with Long- and Short-term User Representations)

#  Based on paper:     Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu and Xing Xie:
#     Neural News Recommendation with Long- and Short-term User Representations, ACL 2019.                 #
#  Adapted from implementatio by Microsoft Recommenders:                                                   #
# https://github.com/recommenders-team/recommenders/blob/main/recommenders/models/newsrec/models/lstur.py  #                                                                               #                                                #                                    #
#############################################################################################################
# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
from ..recommender import Recommender
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from cornac.utils.newsrec_utils.newsrec_utils import NewsRecUtil


import numpy as np
from cornac.utils.newsrec_utils.layers import (
    AttLayer2,
    ComputeMasking,
    OverwriteMasking,
)
import json
from tqdm.auto import tqdm
import re
import random


class LSTUR(Recommender):
    """LSTUR: 
    Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu and Xing Xie:
    Neural News Recommendation with Long- and Short-term User Representations, ACL 2019

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
    """

    def __init__(self, wordEmb_file = None,
                 wordDict_file = None,
                  newsTitle_file=None,
                userHistory=None,
                title_size=30,
                  word_emb_dim=300,
                   history_size=50,
                   gru_unit = 400 ,
                    window_size = 3 ,
                    cnn_activation = "relu",
                    filter_num = 400,
                    name = "LSTUR", 
                    type = "ini",
                    npratio=4,
                    dropout=0.2,
                    attention_hidden_dim=200,
                    learning_rate=0.0001,
                    epochs=5,
                    batch_size=32,

                    trainable=True,
                    verbose=True,
                    seed=42, 
                    word2vec_embedding=None,  # Add embedding for loading directly
                    word_dict=None,
                    news_title=None, 
                    **kwargs):
        """Initialization steps for LSTUR.
        Compared with the BaseModel, LSTUR need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            iterator_creator_train (object): LSTUR data loader class for train data.
            iterator_creator_test (object): LSTUR data loader class for test and validation data
        """
        Recommender.__init__(
            self, name=name, trainable=trainable, verbose=verbose, **kwargs)
        self.seed = seed 
        tf.random.set_seed(seed)
        np.random.seed(seed)

        if word2vec_embedding is not None:
            self.word2vec_embedding = word2vec_embedding  # Load directly from params
        else:
            self.word2vec_embedding = self._init_embedding(wordEmb_file)

        if word_dict is not None:
            self.word_dict = word_dict  # Load directly from params
        else:
            self.word_dict = self.load_dict(wordDict_file)
        
        if news_title is not None:
            self.news_title = news_title  # Load directly from params
        else:
            self.news_title = self.load_dict(newsTitle_file)
        
         # Use kwargs to handle flexible attributes
        self.train_set = kwargs.get("train_set", None)

        self.news_title_index = kwargs.get("news_title_index", None)
        self.news_index_map = kwargs.get("news_index_map", None)
        self.userHistory = userHistory

        self.word_emb_dim = word_emb_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        # self.epochs = epochs
        # self.batch_size = batch_size
        self.title_size = title_size
        self.history_size = history_size
        # self.head_num = head_num
        # self.head_dim = head_dim
        self.npratio = npratio
        self.attention_hidden_dim = attention_hidden_dim
        self.gru_unit = gru_unit 
        self.window_size = window_size 
        self.cnn_activation = cnn_activation
        self.filter_num = filter_num
        self.type = type

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

        ## set News recommendation utils
        # self.news_organizer = NewsRecUtil(news_title =self.news_title, word_dict = self.word_dict,
        #                              impressionRating = self.impressionRating, user_history= self.userHistory,
        #                              history_size = self.history_size,  title_size = self.title_size)
 
        # session_conf = tf.ConfigProto()
        # session_conf.gpu_options.allow_growth = True
        # sess = tf.Session(config=session_conf)
         ## set News recommendation utils
        







    def load_dict(self, file_path):
        """load json file

        Args:
            file path (str): file path

        Returns:
            object:  dictionary
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained glove embeddings file path.

        Returns:
            numpy.ndarray: A constant numpy array.
        """

        return np.load(file_path)


    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        return [batch_data["clicked_title_batch"], batch_data["user_index_batch"]]

    def _get_news_feature_from_iter(self, batch_data):
        return batch_data["candidate_title_batch"]

    def _build_graph(self):
        """Build LSTUR model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_lstur()
        return model, scorer

    def _build_userencoder(self, titleencoder, type="ini"):
        """The main function to create user encoder of LSTUR.

        Args:
            titleencoder (object): the news encoder of LSTUR.

        Return:
            object: the user encoder of LSTUR.
        """
        his_input_title = keras.Input(
            shape=(self.history_size, self.title_size), dtype="int32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        user_embedding_layer = keras.layers.Embedding(
            len(set(self.train_set.uir_tuple[0])),
            self.gru_unit,
            trainable=True,
            embeddings_initializer="zeros",
        )

        long_u_emb = keras.layers.Reshape((self.gru_unit,))(
            user_embedding_layer(user_indexes)
        )
        click_title_presents = keras.layers.TimeDistributed(titleencoder)(his_input_title)

        if type == "ini":
            user_present = keras.layers.GRU(
                self.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(
                keras.layers.Masking(mask_value=0.0)(click_title_presents),
                initial_state=[long_u_emb],
            )
        elif type == "con":
            short_uemb = keras.layers.GRU(
                self.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(keras.layers.Masking(mask_value=0.0)(click_title_presents))

            user_present = keras.layers.Concatenate()([short_uemb, long_u_emb])
            user_present = keras.layers.Dense(
                self.gru_unit,
                bias_initializer=keras.initializers.Zeros(),
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            )(user_present)

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of LSTUR.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of LSTUR.
        """
        sequences_input_title = keras.Input(shape=(self.title_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = keras.layers.Dropout(self.dropout)(embedded_sequences_title)
        y = keras.layers.Conv1D(
            self.filter_num,
            self.window_size,
            activation=self.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        print(y)
        y = keras.layers.Dropout(self.dropout)(y)
        y = keras.layers.Masking()(
            OverwriteMasking()([y, ComputeMasking()(sequences_input_title)])
        )
        pred_title = AttLayer2(self.attention_hidden_dim, seed=self.seed)(y)
        print(pred_title)
        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_lstur(self):
        """The main function to create LSTUR's logic. The core of LSTUR
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        his_input_title = keras.Input(
            shape=(self.history_size, self.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(self.npratio + 1, self.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                self.title_size,
            ),
            dtype="int32",
        )
        pred_title_reshape = keras.layers.Reshape((self.title_size,))(pred_input_title_one)
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        embedding_layer = keras.layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder, type=self.type)
        self.newsencoder = titleencoder

        user_present = self.userencoder([his_input_title, user_indexes])
        news_present = keras.layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_reshape)

        preds = keras.layers.Dot(axes=-1)([news_present, user_present])
        preds = keras.layers.Activation(activation="softmax")(preds)

        pred_one = keras.layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = keras.layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([user_indexes, his_input_title, pred_input_title], preds)
        scorer = keras.Model(
            [user_indexes, his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer
    
    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.


        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        train_input, train_label = self._get_input_label_from_iter(
            train_batch_data)
        rslt = self.model.train_on_batch(train_input, train_label)
        return rslt

    
    def fit(self, train_set, val_set=None):
        """Fit the model with train_file.

        Parameters:
        -----------
            train_set: training data set.
            val_set : validation set.

        Returns:
        -----------
            object: An instance of self.
        """

        
        Recommender.fit(self, train_set, val_set)

        self.train_set = train_set
        self.val_set = val_set

        # impression rating from u-i-r

        # Initialize dictionaries to store positive and negative ratings
        ratings_data = {"positive_rating": {}, "negative_rating": {}}
        

        user_indices, item_indices, rating_values = self.train_set.uir_tuple

        ratings_data = {"positive_rating": {}, "negative_rating": {}}

        for user_idx, item_idx, rating in zip(user_indices, item_indices, rating_values):
            if rating > 0:
                if user_idx not in ratings_data["positive_rating"]:
                    ratings_data["positive_rating"][user_idx] = []
                ratings_data["positive_rating"][user_idx].append(item_idx)
            else:
                if user_idx not in ratings_data["negative_rating"]:
                    ratings_data["negative_rating"][user_idx] = []
                ratings_data["negative_rating"][user_idx].append(item_idx)



        self.news_organizer = NewsRecUtil(news_title =self.news_title, word_dict = self.word_dict,
                                     impressionRating = ratings_data, user_history= self.userHistory,
                                     history_size = self.history_size,  title_size = self.title_size)

        # Configure GPU settings
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(f"GPU memory growth setting failed: {e}")


        # Build model on GPU
        # with tf.device('/GPU:1'):
        self.model, self.scorer = self._build_graph()
        self.model.compile(
        loss="categorical_crossentropy",
        optimizer= keras.optimizers.Adam(learning_rate=self.learning_rate)  
    )
        
        self.loss_log = []  # Store the loss values over epochs
        # self.click_title_all_users = {}
        for epoch in range(1, self.epochs + 1):
            step = 0
            self.current_epoch = epoch
            epoch_loss = 0
            tqdm_util = tqdm(
                self.news_organizer.load_data_from_file(train_set, self.npratio,self.batch_size), desc=f"Epoch {epoch}",
                leave=False  # Removes stale progress bars
            )


            # tqdm_util = tqdm(
            #     self.news_organizer.load_data_from_file(train_set, self.npratio,self.batch_size,self.click_title_all_users), desc=f"Epoch {epoch}"
            # )
            for batch_data_input in tqdm_util:
                # print(f"batch_data_input: {batch_data_input}")
                step_result = self.train(batch_data_input)
                step_data_loss = step_result

                epoch_loss += step_data_loss
                step += 1
                # show step every 10 steps
                if step % 100 == 0:
                    tqdm_util.update(100)
                    tqdm_util.set_postfix(
                        total_loss=f"{epoch_loss / step:.4f}", data_loss=f"{step_data_loss:.4f}"
                        )
                    # tqdm_util.set_description(
                    #     "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                    #         step, epoch_loss / step, step_data_loss
                    #     )
                    # )
            tqdm_util.close()
            avg_epoch_loss = epoch_loss / step
            self.loss_log.append({"epoch": epoch, "loss": avg_epoch_loss})

        # self.save_loss_log(self.loss_log, filename="training_loss_log_ebnerd_lstur_npratio6.csv")
        self.save_loss_log(self.loss_log, filename="training_loss_log_lstur.csv")
        return self
    
    def save_loss_log(self, loss_log, filename="training_loss_log.csv"):
        """Save the training loss log to a CSV or JSON file using Pandas."""

        df = pd.DataFrame(loss_log)

        if filename.endswith(".csv"):
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Training loss log saved as CSV at {filename}")

        elif filename.endswith(".json"):
            # Save to JSON
            # Use line-delimited JSON
            df.to_json(filename, orient="records", lines=True)
            print(f"Training loss log saved as JSON at {filename}")

        else:
            raise ValueError("Unsupported file format. Use .csv or .json.")
    
    def score(self, user_idx, item_idx=None, **kwargs):
        """Predict the scores/ratings of a user for items.

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
        # self.item_idx2id = kwargs.get("item_idx2id", None)
    
        # self.user_idx2id =  kwargs.get("user_idx2id", None)
        self.item_idx2id = {v: k for k, v in self.iid_map.items()} # cornac item ID : raw item ID
        self.user_idx2id = {v: k for k, v in self.uid_map.items()} # cornac user ID : raw user ID

        if not hasattr(self, "news_organizer"):
            raise AttributeError("News organizer not found. Please provide training data and fit the model first.")


        if not hasattr(self.news_organizer, "news_title_index") or self.news_organizer.news_title_index is None:
            self.news_organizer.init_news(self.news_title)

        if item_idx is None:
            item_idx = np.arange(self.total_items)


        # Handle specific item or items
        if isinstance(item_idx, (int, np.integer)):
            # Single item case, convert to list for consistent processing
            item_idx = [item_idx]
        elif not isinstance(item_idx, (list, np.ndarray)):
            raise Exception(
                "item_idx should be an int, list, or numpy array")


        batch_size = 256  # Define batch size
        candidate_title_indexes = []
        click_title_indexes = []
        user_indexes = []
        # Get user's click history or handle unknown users
        if user_idx in self.news_organizer.click_title_all_users:
            click_title_index = self.news_organizer.click_title_all_users[user_idx]
        else:
            # Handle unknown user
            raw_UID = self.user_idx2id[user_idx]
            raw_IID = self.userHistory[raw_UID]
            his_for_user = self.news_organizer.process_history_news_title(raw_IID, self.history_size)
            click_title_index = his_for_user
                
        # Collect all candidate titles for the batch
        for idx in item_idx:
            item_raw_id = self.item_idx2id[idx]
            candidate_title_index = np.array(
                [self.news_organizer.news_title_index[self.news_organizer.news_index_map[item_raw_id]]]
            )

            candidate_title_indexes.append(candidate_title_index)
            click_title_indexes.append(click_title_index)
            user_indexes.append(user_idx)
    


        # Convert lists to numpy arrays
        candidate_title_index_batch = np.asarray(candidate_title_indexes, dtype=np.int64)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        user_index_batch =  np.asarray(user_indexes, dtype=np.int32)



        total_items = candidate_title_index_batch.shape[0]

        # Store predictions
        all_predictions = []

        # Process in batches of 1024
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)  # End index for the batch
            batch_user_index = user_index_batch[start:end]
            batch_candidate_title_index = candidate_title_index_batch[start:end]
            batch_click_title_index = click_title_index_batch[start:end]



            batch_prediction = self.scorer.predict_on_batch(
                [batch_user_index, batch_click_title_index, batch_candidate_title_index]
            )

            all_predictions.append(batch_prediction)
            
        # Concatenate all batch predictions into a single array
        final_predictions = np.concatenate(all_predictions, axis=0)
        # print(f"predictions: {final_predictions}")

        return final_predictions.ravel()

        
