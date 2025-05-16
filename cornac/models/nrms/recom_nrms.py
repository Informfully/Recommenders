#############################################################################################################
# Model name:      Neural News Recommendation with Multi-Head Self-Attention                               #
#  Based on paper: Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie,                  #
#  "Neural News Recommendation with Multi-Head Self-Attention,"                                            #
# Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing                   #
# and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP).                #
#  Adapted from implementatio by Microsoft Recommenders:                                                   #
# https://github.com/recommenders-team/recommenders/blob/main/recommenders/models/newsrec/models/nrms.py   #
# DESCRIPTION:  The `NRMS` class implements the Neural News Recommendation model using multi-head          #
#               self-attention mechanisms for personalized news recommendations. It supports training      #
#               and scoring, with  user and news encoders.                                                 #
#############################################################################################################

from tqdm.auto import trange
from ..recommender import Recommender
# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.compat.v1 import keras
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers
from cornac.utils.newsrec_utils.layers import AttLayer2, SelfAttention
from cornac.utils.newsrec_utils.newsrec_utils import NewsRecUtil
import numpy as np
import pickle
import os
from tqdm.auto import tqdm
import random
import re
import json
import os
import pandas as pd


class NRMS(Recommender):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP).
    """

    def __init__(
        self,
        wordEmb_file=None,  # Allow None for loading from saved params
        wordDict_file=None,
        newsTitle_file=None,
        userHistory=None,
        title_size=30,
        word_emb_dim=300,
        history_size=50,
        name="NRMS",
        npratio=4,
        dropout=0.2,
        attention_hidden_dim=200,
        head_num=20,
        head_dim=20,
        learning_rate=0.0001,
        epochs=5,
        batch_size=32,
        trainable=True,
        verbose=True,
        seed=42,
        word2vec_embedding=None,  # Add embedding for loading directly
        word_dict=None,  # Add word_dict for loading directly
        news_title=None,  # Add news_title for loading directly
        **kwargs
    ):
        """
        Initializes the NRMS class.
        Parameters:
        --------------
        wordEmb_file (str): Path to the file containing the pretrained word embeddings matrix in numpy format.
            Each row corresponds to a word vector, and the row index corresponds to the word's ID.
        wordDict_file (str): Path to a JSON file that maps words to their corresponding indices in the word embedding matrix.
        newsTitle_file (str): Path to a JSON file that maps raw news IDs (e.g., 'N1') to their corresponding titles.
        userHistory (dictionary): maps raw user id to list of history item (raw) ids. For example: {"U10002": ["N63889", "N1]}.
        impressionRating (dict, optional): A dictionary containing user interaction data with news items, structured as:
        {
            "positive_rating": {
                user_id1: [news_id1, news_id2, ...],  # List of positively interacted news IDs for user_id1
                user_id2: [news_id3, news_id4, ...],
                ...
            },
            "negative_rating": {
                user_id1: [news_id5, news_id6, ...],  # List of negatively interacted news IDs for user_id1
                user_id2: [news_id7, news_id8, ...],
                ...
            }
        }
        This is used for generating training labels, negative sampling, and constructing input batches for the model.

        title_size (int, optional): The maximum number of words considered in each news title. Defaults to 30.
        word_emb_dim (int, optional): Dimensionality of word embeddings (size of each word vector). Defaults to 300.
                        This dimension should correspond to the word embeddings matrix's dimensionality.
        history_size (int, optional): The number of previously read articles considered for user modeling. Defaults to 50.
        name (str, optional): Name of the model instance. Defaults to "NRMS".
        npratio (int, optional): Ratio of negative to positive samples used for training (negative sampling). Defaults to 4.
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
        attention_hidden_dim (int, optional): Dimensionality of the hidden layer in the attention mechanism. Defaults to 200.
        head_num (int, optional): Number of attention heads in the multi-head self-attention mechanism. Defaults to 20.
        head_dim (int, optional): Dimensionality of each attention head. Defaults to 20.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
        epochs (int, optional): Number of epochs to train the model. Defaults to 5.
        batch_size (int, optional): Batch size used for training. Defaults to 32.
        trainable (bool, optional): If True, the model parameters are trainable. Defaults to True.
        verbose (bool, optional): If True, enables verbose logging during training and initialization. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

        word2vec_embedding (numpy.ndarray, optional): Preloaded word embeddings as a numpy array. If provided, it will be used directly.
        word_dict (dict, optional): Preloaded word-to-index mapping. If provided, it will be used directly.
        kwargs (dict, optional): Additional keyword arguments for flexible attributes, such as train_set or news indices.


        """
        Recommender.__init__(
            self, name=name, trainable=trainable, verbose=verbose,   **kwargs)
        self.seed = seed
        tf.compat.v1.set_random_seed(seed)

        # Check if embeddings, word_dict, and news titles are provided or need to be loaded
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
        self.word_emb_dim = word_emb_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.title_size = title_size
        self.history_size = history_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.npratio = npratio
        self.attention_hidden_dim = attention_hidden_dim


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
        # with tf.device('/GPU:0'):
        self.model, self.scorer = self._build_graph()
        self.model.compile(loss="categorical_crossentropy",
                            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))

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

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            model: object: a model used to train.
            scorer: object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        # hparams = self.hparams
        his_input_title = keras.Input(
            shape=(self.history_size, self.title_size), dtype="int32"
        )

        click_title_presents = layers.TimeDistributed(
            titleencoder)(his_input_title)
        y = SelfAttention(self.head_num, self.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(
            self.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        sequences_input_title = keras.Input(
            shape=(self.title_size,), dtype="int32")

        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(self.dropout)(embedded_sequences_title)
        y = SelfAttention(self.head_num, self.head_dim,
                          seed=self.seed)([y, y, y])
        y = layers.Dropout(self.dropout)(y)
        pred_title = AttLayer2(self.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(sequences_input_title,
                            pred_title, name="news_encoder")
        return model

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning.

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]

        input_label = batch_data["labels"]

        return input_feat, input_label

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
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
        pred_title_one_reshape = layers.Reshape(
            (self.title_size,))(pred_input_title_one)

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)

        self.userencoder = self._build_userencoder(titleencoder)

        self.newsencoder = titleencoder

        user_present = self.userencoder(his_input_title)
        news_present = layers.TimeDistributed(
            self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([his_input_title, pred_input_title], preds)
        scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)

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

        ## set News recommendation utils
        self.news_organizer = NewsRecUtil(news_title =self.news_title, word_dict = self.word_dict,
                                     impressionRating = ratings_data, user_history= self.userHistory,
                                     history_size = self.history_size,  title_size = self.title_size)
        

        self.loss_log = []  # Store the loss values over epochs

        # Early stopping parameters
        best_loss = float("inf")  # Initialize with a large value
        patience_counter = 0  # To track epochs without improvement
        patience = 5  # Number of epochs to wait for improvement before stopping
        min_delta = 0.004  # Minimum change to qualify as an improvement

        log_every_n_steps = 100  # Log every 100 steps
        # self.click_title_all_users = {}
        for epoch in range(1, self.epochs + 1):
            step = 0
            self.current_epoch = epoch
            epoch_loss = 0

            tqdm_util = tqdm(
                self.news_organizer.load_data_from_file(train_set, self.npratio,self.batch_size), desc=f"Epoch {epoch}",
                leave=False , # Removes stale progress bars
                smoothing=0,  # less flicker
                ncols=100
            )

            for batch_data_input in tqdm_util:
                # print(f"batch_data_input:{batch_data_input}")

                step_result = self.train(batch_data_input)
                step_data_loss = step_result

                epoch_loss += step_data_loss
                step += 1
                # show step every 100 steps
                if step % log_every_n_steps == 0:
                    tqdm_util.update(log_every_n_steps)
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


        self.save_loss_log(self.loss_log, filename="training_loss_log_nrms.csv")
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

        # Convert lists to numpy arrays
        candidate_title_index_batch = np.asarray(candidate_title_indexes, dtype=np.int64)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)

        # print(f"candidate_title_index_batch:{candidate_title_index_batch}")
        # print(f"click_title_index_batch:{click_title_index_batch}")

        # Get total number of items
        total_items = candidate_title_index_batch.shape[0]

        # Store predictions
        all_predictions = []

        # Process in batches of 1024
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)  # End index for the batch

            batch_candidate_title_index = candidate_title_index_batch[start:end]
            batch_click_title_index = click_title_index_batch[start:end]

            # print(f"Processing batch: {start}-{end}, Shape: {batch_candidate_title_index.shape}")

            batch_prediction = self.scorer.predict_on_batch(
                [batch_click_title_index, batch_candidate_title_index]
            )

            all_predictions.append(batch_prediction)

        # Concatenate all batch predictions into a single array
        final_predictions = np.concatenate(all_predictions, axis=0)

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
            "head_num": self.head_num,
            "head_dim": self.head_dim,
            "npratio": self.npratio,
            "attention_hidden_dim": self.attention_hidden_dim,
            "seed": self.seed,
            "userHistory": self.userHistory,  # Save user history
            "news_title": self.news_title,  # News titles dictionary
            "word_dict": self.word_dict,  # Word dictionary
            "word2vec_embedding": self.word2vec_embedding,  # Word embeddings matrix
        }

        # Optionally save train_set, news_title_index, and news_index_map if they exist
        if hasattr(self, 'train_set') and self.train_set is not None:
            params["train_set"] = self.train_set
   

        # Save the parameters as a pickle file
        params_file_path = os.path.join(save_dir, "nrms_params.pkl")
        with open(params_file_path, "wb") as f:
            pickle.dump(params, f)

        # Save only the model weights (not the architecture)
        model_file_path = os.path.join(save_dir, "nrms_model_weights.h5")
        self.model.save_weights(model_file_path)

        # Save only the scorer model weights
        scorer_file_path = os.path.join(save_dir, "nrms_scorer_weights.h5")
        self.scorer.save_weights(scorer_file_path)

        print(f"Model saved successfully to {save_dir}")

    @classmethod
    def load_nrms(cls, save_dir):
        """
        Load the NRMS model, including its trained weights and internal parameters, from the specified directory.


        Parameters:
            save_dir (str): The directory where the model and parameters are saved.

        Returns:
            NRMS: The loaded NRMS model object.
        """
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            raise FileNotFoundError(
                f"The directory {save_dir} does not exist.")

        # Load the saved parameters (such as hyperparameters, word embeddings, dictionaries, etc.)
        params_file_path = os.path.join(save_dir, "nrms_params.pkl")
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
            head_num=params["head_num"],
            head_dim=params["head_dim"],
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

        # Load the saved model weights
        model_file_path = os.path.join(save_dir, "nrms_model_weights.h5")
        scorer_file_path = os.path.join(save_dir, "nrms_scorer_weights.h5")

        if not os.path.exists(model_file_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_file_path}")

        if not os.path.exists(scorer_file_path):
            raise FileNotFoundError(
                f"Scorer weights not found at {scorer_file_path}")

        # Load the weights into the models
        model.model.load_weights(model_file_path)
        model.scorer.load_weights(scorer_file_path)

        print(f"Model loaded successfully from {save_dir}")

        return model
