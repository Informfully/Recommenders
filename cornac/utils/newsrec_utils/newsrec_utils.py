import random
import re
import numpy as np
import json
import pandas as pd

class NewsRecUtil:
    def __init__(self,news_title = None, word_dict = None, impressionRating = None, user_history = None, history_size=50, title_size = 30 ):
        self.hisory_size = history_size
        self.title_size = title_size
        self.impressionRating = impressionRating
        self.user_history  = user_history
        self.news_title = news_title
        self.word_dict = word_dict
        self.click_title_all_users = {}
 

    def newsample(self, news, ratio):
        """Sample ratio samples from news list.
        If length of news is less than ratio, pad zeros.

        Parameters:
        -------------
            news (list): input news list, indexes
            ratio (int): sample number

        Returns:
        -------------
            list: output of sample list.
        """
        if ratio > len(news):
            return news + [0] * (ratio - len(news))
        else:
            return random.sample(news, ratio)


    def load_data_from_file(self, train_set, npratio, batch_size):
        """
            Prepares and yields batches of training data from the given train_set, 
            mapping user behavior (clicks and non-clicks) to news titles and labels.

            This function processes the training dataset by extracting the user interactions 
            with news items (clicks and non-clicks) and prepares batches of data suitable 
            for training the NRMS model. It yields the data in batches based on the specified 
            batch size.

            Parameters:
            -------------
                train_set (object): The training dataset containing user interactions in 
                CSR matrix format. Each row represents a user, with columns representing 
                news articles, and values indicating whether the user clicked on the article.
                - clicked_article_titles_dict (dict): Dictionary to store users' clicked article titles.

            Yields:
            -------------
                batch (before calling _convert_data function):
                    - label_list: List of labels indicating clicked news (1) and non-clicked news (0).
                    - user_indexes: List of user indices corresponding to the batch.
                    - candidate_title_indexes: List of indices of candidate news titles (clicked and non-clicked).
                    - click_title_indexes: List of indices of news titles that users have previously clicked.

            The process consists of the following steps:
                1. Initialize user history and impression logs (news titles clicked by users) if not already done.
                2. Retrieve positive and negative interaction items (news) for each user.
                3. For each clicked (positive) item, sample negative items based on the negative-positive ratio (npratio).
                4. Convert these positive and negative items into sequential word indices representing their news titles.
                5. Retrieve and pad/truncate the user's history of clicked items to a fixed size.
                6. Accumulate the processed data into batches, yielding each batch once the batch size is reached.
                7. If any remaining data is left after the final batch, it yields the remaining data.
        """

        if not hasattr(self, "news_title_index") or self.news_title_index is None:
            print("init news")
            self.init_news( self.news_title)
        
        # item od to Cornac ID
        self.item_id2idx = {k: v for k, v in train_set.iid_map.items()}
         # Cornac item ID to original item ID
        self.item_idx2id = {v: k for k, v in train_set.iid_map.items()}

        # original user ID to Cornac user ID
        self.user_id2idx = {k: v for k, v in train_set.uid_map.items()}
        # Cornac user ID to original user ID
        self.user_idx2id = {v: k for k, v in train_set.uid_map.items()}

        label_list = []

        user_indexes = []
        candidate_title_indexes = []
        click_title_indexes = []

        cnt = 0
        if not hasattr(train_set, "uir_tuple"):
            raise ValueError("train_set does not contain the required 'uir_tuple' attribute.")

        # train_set_user_indices = set(train_set.uir_tuple[0])
        train_set_user_indices = list(set(train_set.uir_tuple[0]))
        np.random.shuffle(train_set_user_indices)

        for user_idx in train_set_user_indices:
            raw_UID = self.user_idx2id[user_idx]

            raw_IID = self.user_history[raw_UID]
            his_for_user = self.process_history_news_title(raw_IID, self.hisory_size)

            if user_idx in self.impressionRating["positive_rating"] and user_idx in self.impressionRating["negative_rating"]:
                train_pos_items = self.impressionRating["positive_rating"][user_idx]

                train_neg_items = self.impressionRating["negative_rating"][user_idx]

                if len(train_pos_items) > 0:
                    for p in train_pos_items:
                        candidate_title_index = []
                        user_index = []
                        label = [1] + [0] * npratio
                        user_index.append(user_idx)
                        n = self.newsample(train_neg_items, npratio)
                        # Convert `p` and `n` to sequential indices using `news_index_map`
                        candidate_keys = [p] + n
                        raw_item_ids = [self.item_idx2id[k] for k in candidate_keys]
                        candidate_title_index = np.array(
                            [self.news_title_index[self.news_index_map[key]] for key in raw_item_ids])

                        click_title_index = his_for_user
                        self.click_title_all_users[user_idx] = click_title_index

                        candidate_title_indexes.append(candidate_title_index)
                        click_title_indexes.append(click_title_index)
                        user_indexes.append(user_index)
                        label_list.append(label)
                        cnt += 1
                        

            # cnt += 1
                        if cnt >= batch_size:
                            yield self._convert_data(
                                label_list,
                                user_indexes,
                                candidate_title_indexes,
                                click_title_indexes,
                            )
                            label_list = []
                            user_indexes = []
                            candidate_title_indexes = []
                            click_title_indexes = []
                            cnt = 0

        if cnt > 0:
            yield self._convert_data(
                label_list,
                user_indexes,
                candidate_title_indexes,
                click_title_indexes,
            )


    def _convert_data(
            self,
            label_list,
            user_indexes,
            candidate_title_indexes,
            click_title_indexes,
        ):
        """Convert data into numpy arrays for further model operation.

        Parameters:
            label_list (list): a list of ground-truth labels.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        labels = np.asarray(label_list, dtype=np.float32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
        click_title_index_batch = np.asarray(
            click_title_indexes, dtype=np.int64)
        return {
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "labels": labels,
        }

    def map_news_titles_to_Cornac_internal_ids(self, train_set, news_original_id_to_news_title):

        # original item ID to Cornac item ID
        self.item_id2idx = {k: v for k, v in train_set.iid_map.items()}
        # Cornac item ID to original item ID
        self.item_idx2id = {v: k for k, v in train_set.iid_map.items()}

        # original user ID to Cornac user ID
        self.user_id2idx = {k: v for k, v in train_set.uid_map.items()}
        # Cornac user ID to original user ID
        self.user_idx2id = {v: k for k, v in train_set.uid_map.items()}
        feature_map = {}
        for key, value in news_original_id_to_news_title.items():
            if key in self.item_id2idx:
                idx = self.item_id2idx[key]
                feature_map[idx] = value
            # feature_map[key] = value
        missing_keys = set(self.item_id2idx.values()) - set(feature_map.keys())


        if not missing_keys:
            print("All keys in item_id2idx are present in feature_map.")
        else:
            print(f"Missing keys in feature_map: {missing_keys}")
            raw_ids = [self.item_idx2id[id0] for id0 in missing_keys]
            print(f"Missing raw item titles: {raw_ids}")

        return feature_map
    

    def process_history_news_title(self, history_raw_IID, history_size):
        """init news information given news file, such as news_title_index.
        Args:
            news_file: path of news file
            history_raw_IID: raw item ids for a user
            history_size: the fixed history size to keep.
        """

        news_title = {}
        # original_UID = self.user_idx2id[user_idx]
        # get user History item ids
        # his_original_IID = self.userHistory[original_UID]

        def pad_or_truncate(sequence, max_length):
            if len(sequence) < max_length:
                # Pad with -1 if the sequence is too short
                return [-1] * (max_length - len(sequence)) + sequence
            else:
                # Truncate the sequence if it's too long
                return sequence[-max_length:]

        history_raw_IID = pad_or_truncate(history_raw_IID, history_size)
        news_json = []
        for i in history_raw_IID:
            if i in self.news_title:
                news_json.append(self.news_title[i])
            elif i == -1:

                news_json.append("")

        news_title = []
        for value in news_json:

            title = self.word_tokenize(value)
            news_title.append(title)

        his_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )
        # total news_title * word size
        for i in range(len(news_title)):
            title = news_title[i]
            for word_index in range(min(self.title_size, len(title))):
                word = title[word_index].lower()
                if word in self.word_dict:
                    his_index[i, word_index] = self.word_dict[word]
        return his_index

    

    def init_news(self, news_title_json):
        """init news information given news file, such as news_title_index.
        Args:
            news_file: path of news file
        """
        news_title = {}
        # news_json = self.map_news_titles_to_Cornac_internal_ids(train_set,
        #     news_title_json)
        news_json = news_title_json

        news_json[-1] = ""
        # Map cornac ID to a sequential index
        self.news_index_map = {key: idx for idx,
                               key in enumerate(news_json.keys())}


        for key, value in news_json.items():
           
            if key == -1:
                news_title[key] = ""
            else:
                title = self.word_tokenize(value)
                news_title[key] = title
            # if key > -1:
            #     title = self.word_tokenize(value)
            #     news_title[key] = title
            # elif key == -1:
            #     news_title[key] = ""
        # for "", news_title[-1] = [] empty list

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )
        for key, title in news_title.items():
            mapped_index = self.news_index_map[key]
            for word_index in range(min(self.title_size, len(title))):
                word = title[word_index].lower()
                if word in self.word_dict:
                    self.news_title_index[mapped_index,
                                          word_index] = self.word_dict[word]
        # print(f"self.news_index_map:{self.news_index_map}")  
        # print(f"self.news_title_index:{self.news_title_index}")   

    def word_tokenize(self, sent):
        """Split sentence into word list using regex.
        Parameters:
        ------------
            sent (str): Input sentence

        Return:
        ------------
            list: word list
        """
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []


        

            

