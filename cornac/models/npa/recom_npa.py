# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


from ..recommender import Recommender
import random
import re
import json
import tensorflow as tf
from tensorflow.compat.v1 import keras
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers
import numpy as np
from cornac.utils.newsrec_utils.layers import PersonalizedAttentivePooling
from cornac.utils.newsrec_utils.newsrec_utils import NewsRecUtil
import pandas as pd
from tqdm.auto import tqdm
# tf.compat.v1.disable_eager_execution()
import os
import pickle

class NPA(Recommender):
    """NPA model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie:
    NPA: Neural News Recommendation with Personalized Attention, KDD 2019, ADS track.

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, 
                 wordEmb_file=None,  # Allow None for loading from saved params
                wordDict_file=None,
                newsTitle_file=None,
                userHistory=None,    
                title_size=30,
                word_emb_dim=300,
                user_emb_dim = 100, 
                history_size=50,
                name="NPA",
                npratio=4,
                dropout=0.2,
                attention_hidden_dim=200,
                window_size = 3 ,
                cnn_activation = "relu",
                filter_num = 400,

                learning_rate=0.0001,
                epochs=5,
                batch_size=32,
                trainable=True,
                verbose=True,
                seed=42,
                word2vec_embedding=None,  # Add embedding for loading directly
                word_dict=None,  # Add word_dict for loading directly
                news_title=None,  # Add news_title for loading directly
                **kwargs):
        """Initialization steps for NPA.
        Compared with the BaseModel, NPA need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train (object): NPA data loader class for train data.
            iterator_creator_test (object): NPA data loader class for test and validation data
        """
        Recommender.__init__(
            self, name=name, trainable=trainable, verbose=verbose,  **kwargs)
        self.seed = seed
        tf.compat.v1.set_random_seed(seed)
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

        # self.news_title_index = kwargs.get("news_title_index", None)
        # self.news_index_map = kwargs.get("news_index_map", None)
        self.userHistory = userHistory
        self.user_emb_dim = user_emb_dim

        self.word_emb_dim = word_emb_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        self.title_size = title_size
        self.history_size = history_size
        self.npratio = npratio
        self.attention_hidden_dim = attention_hidden_dim
        self.window_size = window_size 
        self.cnn_activation = cnn_activation
        self.filter_num = filter_num

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

    def _build_graph(self):
        """Build NPA model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_npa()
        return model, scorer

    def _build_userencoder(self, titleencoder, user_embedding_layer):
        """The main function to create user encoder of NPA.

        Args:
            titleencoder (object): the news encoder of NPA.

        Return:
            object: the user encoder of NPA.
        """
        

        his_input_title = keras.Input(
            shape=(self.history_size, self.title_size), dtype="int32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_id = layers.Reshape((1, 1))(user_indexes)
        repeat_uids = layers.Concatenate(axis=-2)([nuser_id] *  self.history_size)
        his_title_uid = layers.Concatenate(axis=-1)([his_input_title, repeat_uids])

        click_title_presents = layers.TimeDistributed(titleencoder)(his_title_uid)

        u_emb = layers.Reshape((self.user_emb_dim,))(
            user_embedding_layer(user_indexes)
        )
        user_present = PersonalizedAttentivePooling(
            self.history_size,
            self.filter_num,
            self.attention_hidden_dim,
            seed=self.seed,
        )([click_title_presents, layers.Dense(self.attention_hidden_dim)(u_emb)])

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer, user_embedding_layer):
        """The main function to create news encoder of NPA.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NPA.
        """
        sequence_title_uindex = keras.Input(
            shape=(self.title_size + 1,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : self.title_size])(
            sequence_title_uindex
        )
        user_index = layers.Lambda(lambda x: x[:, self.title_size :])(
            sequence_title_uindex
        )

        u_emb = layers.Reshape((self.user_emb_dim,))(
            user_embedding_layer(user_index)
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(self.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            self.filter_num,
            self.window_size,
            activation=self.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(self.dropout)(y)

        pred_title = PersonalizedAttentivePooling(
            self.title_size,
            self.filter_num,
            self.attention_hidden_dim,
            seed=self.seed,
        )([y, layers.Dense(self.attention_hidden_dim)(u_emb)])

        # pred_title = Reshape((1, feature_size))(pred_title)
        model = keras.Model(sequence_title_uindex, pred_title, name="news_encoder")
        return model

    def _build_npa(self):
        """The main function to create NPA's logic. The core of NPA
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and predict.
        """

        his_input_title = keras.Input(
            shape=(self.history_size, self.title_size), dtype="int32"
        )
        # print(f"history input:{self.history_size}")
        pred_input_title = keras.Input(
            shape=(self.npratio + 1, self.title_size), dtype="int32"
        )
        # print(f"self.npratio input:{self.npratio}")
        # print(f"self.title_size input:{self.title_size}")
        pred_input_title_one = keras.Input(
            shape=(
                1,
                self.title_size,
            ),
            dtype="int32",
        )

        pred_title_one_reshape = layers.Reshape((self.title_size,))(
            pred_input_title_one
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_index = layers.Reshape((1, 1))(user_indexes)
        repeat_uindex = layers.Concatenate(axis=-2)(
            [nuser_index] * (self.npratio + 1)
        )
        pred_title_uindex = layers.Concatenate(axis=-1)(
            [pred_input_title, repeat_uindex]
        )
        pred_title_uindex_one = layers.Concatenate()(
            [pred_title_one_reshape, user_indexes]
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        # print(f"self.word2vec_embedding.shape[0] input:{ self.word2vec_embedding.shape[0]}")
       
        user_embedding_layer = layers.Embedding(
            len(set(self.train_set.uir_tuple[0])),
            self.user_emb_dim,
            trainable=True,
            embeddings_initializer="zeros",
        )
        # print(f"self.user_embedding_layer input:{len(set(self.train_set.uir_tuple[0]))}")

        titleencoder = self._build_newsencoder(embedding_layer, user_embedding_layer)
        userencoder = self._build_userencoder(titleencoder, user_embedding_layer)
        newsencoder = titleencoder

        user_present = userencoder([his_input_title, user_indexes])

        news_present = layers.TimeDistributed(newsencoder)(pred_title_uindex)
        news_present_one = newsencoder(pred_title_uindex_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

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
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(f"GPU memory growth setting failed: {e}")

        # Build model on GPU
        # with tf.device('/GPU:3'):
        self.model, self.scorer = self._build_graph()
        self.model.compile(loss="categorical_crossentropy",
                            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))

        # self.model, self.scorer = self._build_graph()

        # self.model.compile(loss="categorical_crossentropy",
        #                    optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))


        self.loss_log = []  # Store the loss values over epochs


        # Early stopping parameters
        best_loss = float("inf")  # Initialize with a large value
        patience_counter = 0  # To track epochs without improvement
        patience = 5  # Number of epochs to wait for improvement before stopping
        min_delta = 0.004  # Minimum change to qualify as an improvement


        # self.click_title_all_users = {}
        for epoch in range(1, self.epochs + 1):
            step = 0
            self.current_epoch = epoch
            epoch_loss = 0

            tqdm_util = tqdm(
                self.news_organizer.load_data_from_file(train_set, self.npratio,self.batch_size), desc=f"Epoch {epoch}",
                leave=False  # Removes stale progress bars
            )

            for batch_data_input in tqdm_util:
                # print(f"batch_data_input: {batch_data_input}")
                # print(f"batch_data_input:{batch_data_input}")
                step_result = self.train(batch_data_input)
                step_data_loss = step_result

                epoch_loss += step_data_loss
                step += 1
                # show step every 10 steps
                # if step % 10 == 0:
                #     tqdm_util.set_description(
                #         "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                #             step, epoch_loss / step, step_data_loss
                #         )
                #     )
                if step % 100 == 0:
                    tqdm_util.update(100)
                    tqdm_util.set_postfix(
                        total_loss=f"{epoch_loss / step:.4f}", data_loss=f"{step_data_loss:.4f}"
                        )
            tqdm_util.close()
            avg_epoch_loss = epoch_loss / step
            self.loss_log.append({"epoch": epoch, "loss": avg_epoch_loss})

         # Check if early stopping criteria is met (based on training loss)
            if avg_epoch_loss < best_loss - min_delta:
                best_loss = avg_epoch_loss
                patience_counter = 0  # Reset patience counter if we have improvement
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
                break  # Stop training early

        self.save_loss_log(self.loss_log, filename="training_loss_log_EBNerd_npa.csv")
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
        
        batch_size = 256
        candidate_title_indexes = []
        click_title_indexes = []
        user_indexes = []
        if user_idx in self.news_organizer.click_title_all_users:
            click_title_index = self.news_organizer.click_title_all_users[user_idx]
        else:
            # Handle unknown user
            raw_UID = self.user_idx2id[user_idx]
            raw_IID = self.userHistory[raw_UID]
            his_for_user = self.news_organizer.process_history_news_title(raw_IID, self.history_size)
            click_title_index = his_for_user
        
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

        # print(f"candidate_title_index_batch:{candidate_title_index_batch}")
        # print(f"click_title_index_batch:{click_title_index_batch}")
        # print(f"user_index_batch:{user_index_batch}")

        # Get total number of items
        total_items = candidate_title_index_batch.shape[0]

        # Store predictions
        all_predictions = []

        # Process in batches of 1024
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)  # End index for the batch
            
            batch_user_index = user_index_batch[start:end]
            batch_candidate_title_index = candidate_title_index_batch[start:end]
            batch_click_title_index = click_title_index_batch[start:end]

            # print(f"Processing batch: {start}-{end}, Shape: {batch_candidate_title_index.shape}")

            batch_prediction = self.scorer.predict_on_batch(
                [batch_user_index, batch_click_title_index, batch_candidate_title_index]
            )

            all_predictions.append(batch_prediction)

        # Concatenate all batch predictions into a single array
        
        
        final_predictions = np.concatenate(all_predictions, axis=0)
        # print(f"final_predictions:{final_predictions}")

        return final_predictions.ravel()


    def save(self, save_dir=None):
        """
        Save the NRMS model's parameters, weights, and other necessary objects for future loading.

        Args:
            save_dir (str): The directory where all model components will be saved.
        """
        if save_dir is None:
            raise ValueError(
                "The 'save_dir' argument is required and cannot be None.")

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save model parameters (such as hyperparameters, word embeddings, dictionaries, etc.)
        params = {
            "word_emb_dim": self.word_emb_dim,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "title_size": self.title_size,
            "history_size": self.history_size,
            "window_size": self.window_size,
            "cnn_activation":self.cnn_activation,
            "filter_num":self.filter_num ,
  
            "npratio": self.npratio,
            "attention_hidden_dim": self.attention_hidden_dim,
            "seed": self.seed,
            "userHistory": self.userHistory,  # Save user history
            "user_emb_dim": self.user_emb_dim,
            "news_title": self.news_title,  # News titles dictionary
            "word_dict": self.word_dict,  # Word dictionary
            "word2vec_embedding": self.word2vec_embedding,  # Word embeddings matrix

        }

        # Optionally save train_set, news_title_index, and news_index_map if they exist
        if hasattr(self, 'train_set') and self.train_set is not None:
            params["train_set"] = self.train_set
        
        # print(f"params:{params}")

        # Save the parameters as a pickle file
        params_file_path = os.path.join(save_dir, "npa_params.pkl")
        with open(params_file_path, "wb") as f:
            pickle.dump(params, f)

        # Save only the model weights (not the architecture)
        model_file_path = os.path.join(save_dir, "npa_model_weights.h5")
        self.model.save_weights(model_file_path)

        # Save only the scorer model weights
        scorer_file_path = os.path.join(save_dir, "npa_scorer_weights.h5")
        self.scorer.save_weights(scorer_file_path)

        print(f"Model saved successfully to {save_dir}")

    @classmethod
    def load_npa(cls, save_dir):
        """
        Load the NPA model, including its trained weights and internal parameters, from the specified directory.


        Parameters:
            save_dir (str): The directory where the model and parameters are saved.

        Returns:
            NPA: The loaded NPA model object.
        """
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            raise FileNotFoundError(
                f"The directory {save_dir} does not exist.")

        # Load the saved parameters (such as hyperparameters, word embeddings, dictionaries, etc.)
        params_file_path = os.path.join(save_dir, "npa_params.pkl")
        with open(params_file_path, "rb") as f:
            params = pickle.load(f)

        # Create an NRMS instance and pass the loaded parameters
        model = cls(
            wordEmb_file=None,  # Don't load from file, we will pass the embedding directly
            wordDict_file=None,
            newsTitle_file=None,
            word_emb_dim=params["word_emb_dim"],
            learning_rate=params["learning_rate"],
            dropout=params["dropout"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            title_size=params["title_size"],
            history_size=params["history_size"],
            npratio=params["npratio"],
            attention_hidden_dim=params["attention_hidden_dim"],
            seed=params["seed"],
            userHistory=params["userHistory"],  # Load user history
            # Load embedding from params
            word2vec_embedding=params["word2vec_embedding"],
            word_dict=params["word_dict"],  # Load word_dict from params
            news_title=params["news_title"],  # Load news_title from params
            # Load train_set if available
            train_set=params.get("train_set", None),
            filter_num = params["filter_num"],
            cnn_activation = params["cnn_activation"],
            window_size = params["window_size"],
            user_emb_dim = params["user_emb_dim"]


            # Load news_title_index if available
            # news_title_index=params.get("news_title_index", None),
            # Load news_index_map if available
            # news_index_map=params.get("news_index_map", None)
        )
        if model.train_set is not None:
            model.num_users = model.train_set.num_users
            model.num_items = model.train_set.num_items
            model.uid_map = model.train_set.uid_map
            model.iid_map = model.train_set.iid_map
            model.min_rating = model.train_set.min_rating
            model.max_rating = model.train_set.max_rating
            model.global_mean = model.train_set.global_mean


        

        # Rebuild the model architecture
        model.model, model.scorer = model._build_graph()

        # Compile the model with the stored learning rate
        model.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=model.learning_rate)
        )

        #################
        # Print out the summary of the model you're about to load the weights into
        # print(f"Current model architecture:{model.model.summary()}")
        # print(f"model. filter_num:{model.filter_num}")
        # print(f"model. cnn_activation:{model.cnn_activation}")
        # print(f"model.user_emb_dim :{model.user_emb_dim}")
       
        ##################

        # Load the saved model weights
        model_file_path = os.path.join(save_dir, "npa_model_weights.h5")
        # model_file_path = os.path.join(save_dir, "npa_ckpt")
        scorer_file_path = os.path.join(save_dir, "npa_scorer_weights.h5")


        if not os.path.exists(model_file_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_file_path}")

        if not os.path.exists(scorer_file_path):
            raise FileNotFoundError(
                f"Scorer weights not found at {scorer_file_path}")

        # Load the weights into the models
        model.model.load_weights(model_file_path)
        # model.model.load_weights(model_file_path)
        model.scorer.load_weights(scorer_file_path)

        print(f"Model loaded successfully from {save_dir}")

        return model
    




                

    

