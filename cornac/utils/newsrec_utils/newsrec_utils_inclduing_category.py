import random
import re
import numpy as np
import json
import pandas as pd
from .newsrec_utils import NewsRecUtil

class NewsRecUtil_including_category(NewsRecUtil):
    def __init__(self,news_title = None, word_dict = None, ab_dict = None, 
                 
                 news_vert = None,
                 news_subvert = None,
                 
                 vert_dict = None, subvert_dict = None, impressionRating = None, user_history = None, history_size=50, title_size = 30, body_size = 30 ,
    
                 ):
        super().__init__(news_title = news_title, word_dict = word_dict, impressionRating = impressionRating, user_history = user_history, history_size=history_size, title_size = title_size )

        ## For NAML model
        self.news_ab = ab_dict
        self.body_size = body_size
        self.news_vert =  news_vert
        self.news_subvert = news_subvert
        self.vert_dict= vert_dict
        
        self.subvert_dict = subvert_dict


        self.click_news_ab_all_users = {}
        self.click_news_vert_all_users = {}
        self.click_news_subvert_all_users = {}

       

        


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
                self.init_news(self.news_title, self.news_ab, self.news_vert, self.news_subvert, self.vert_dict, self.subvert_dict )
            
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
            candidate_ab_indexes = []
            candidate_vert_indexes = []
            candidate_subvert_indexes = []
            click_title_indexes = []
            click_ab_indexes = []
            click_vert_indexes = []
            click_subvert_indexes = []

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
                his_ab_for_user = self.process_history_news_ab(raw_IID, self.hisory_size)
                his_vert_for_user = self.process_history_news_vert(raw_IID, self.hisory_size)
                his_subvert_for_user = self.process_history_news_subvert(raw_IID, self.hisory_size)

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
                            candidate_ab_index = np.array(
                                [self.news_ab_index[self.news_index_map[key]] for key in raw_item_ids])
                            candidate_vert_index = np.array(
                                [self.news_vert_index[self.news_index_map[key]] for key in raw_item_ids])
                            candidate_subvert_index = np.array(
                                [self.news_subvert_index[self.news_index_map[key]] for key in raw_item_ids])

                            click_title_index = his_for_user

                            self.click_title_all_users[user_idx] = click_title_index
                            
                            # clicked_article_titles_dict[user_idx] = click_title_index
                            click_ab_index = his_ab_for_user

                            click_vert_index = his_vert_for_user
                            click_subvert_index = his_subvert_for_user

                            self.click_news_ab_all_users[user_idx] = click_ab_index
                            self.click_news_vert_all_users[user_idx] = click_vert_index
                            self.click_news_subvert_all_users[user_idx] = click_subvert_index
                            
                            candidate_title_indexes.append(candidate_title_index)
                            candidate_ab_indexes.append(candidate_ab_index)
                            candidate_vert_indexes.append(candidate_vert_index)
                            candidate_subvert_indexes.append(candidate_subvert_index)
                            click_title_indexes.append(click_title_index)
                            click_ab_indexes.append(click_ab_index)
                            click_vert_indexes.append(click_vert_index)
                            click_subvert_indexes.append(click_subvert_index)
                            user_indexes.append(user_index)
                            label_list.append(label)

                            cnt += 1
                            if cnt >= batch_size:
                                yield self._convert_data(
                                    label_list,
                                    user_indexes,
                                    candidate_title_indexes,
                                    candidate_ab_indexes,
                                    candidate_vert_indexes,
                                    candidate_subvert_indexes,
                                    click_title_indexes,
                                    click_ab_indexes,
                                    click_vert_indexes,
                                    click_subvert_indexes

                                )
                                label_list = []

                                user_indexes = []
                                candidate_title_indexes = []
                                candidate_ab_indexes = []
                                candidate_vert_indexes = []
                                candidate_subvert_indexes = []
                                click_title_indexes = []
                                click_ab_indexes = []
                                click_vert_indexes = []
                                click_subvert_indexes = []
                                cnt = 0
                                

            if cnt > 0:
                yield self._convert_data(
                      label_list,
                        user_indexes,
                        candidate_title_indexes,
                        candidate_ab_indexes,
                        candidate_vert_indexes,
                        candidate_subvert_indexes,
                        click_title_indexes,
                        click_ab_indexes,
                        click_vert_indexes,
                        click_subvert_indexes,
                )


    def _convert_data(
        self,
        label_list,
        user_indexes,
        candidate_title_indexes,
        candidate_ab_indexes,
        candidate_vert_indexes,
        candidate_subvert_indexes,
        click_title_indexes,
        click_ab_indexes,
        click_vert_indexes,
        click_subvert_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            candidate_ab_indexes (list): the candidate news abstarcts' words indices.
            candidate_vert_indexes (list): the candidate news verts' words indices.
            candidate_subvert_indexes (list): the candidate news subverts' indices.
            click_title_indexes (list): words indices for user's clicked news titles.
            click_ab_indexes (list): words indices for user's clicked news abstarcts.
            click_vert_indexes (list): indices for user's clicked news verts.
            click_subvert_indexes (list):indices for user's clicked news subverts.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        labels = np.asarray(label_list, dtype=np.float32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
        candidate_ab_index_batch = np.asarray(candidate_ab_indexes, dtype=np.int64)
        candidate_vert_index_batch = np.asarray(candidate_vert_indexes, dtype=np.int64)
        candidate_subvert_index_batch = np.asarray(
            candidate_subvert_indexes, dtype=np.int64
        )
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_ab_index_batch = np.asarray(click_ab_indexes, dtype=np.int64)
        click_vert_index_batch = np.asarray(click_vert_indexes, dtype=np.int64)
        click_subvert_index_batch = np.asarray(click_subvert_indexes, dtype=np.int64)
        return {
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_ab_batch": click_ab_index_batch,
            "clicked_vert_batch": click_vert_index_batch,
            "clicked_subvert_batch": click_subvert_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_ab_batch": candidate_ab_index_batch,
            "candidate_vert_batch": candidate_vert_index_batch,
            "candidate_subvert_batch": candidate_subvert_index_batch,
            "labels": labels,
        }




    def init_news(self, news_title_json, news_ab_json, news_vert_json, news_subvert_json, vert_dict, subvert_dict ):
        """init news information given news file, such as news_title_index.
        Args:
            news_file: path of news file
        """
        super().init_news(news_title_json)
        # print(f"len news_ab:{len(news_ab)}")
        # print(f"len news_vert:{len(news_vert)}")
        # print(f"len news_subvert:{len(news_subvert)}")
        news_ab = {}
        news_ab_mapped = news_ab_json
        news_ab_mapped[-1] = "" 
        
        news_vert = news_vert_json
        news_vert[-1] = ""
        # news_vert[""] = 0

        news_subvert = news_subvert_json
        news_subvert[-1] = ""
        # news_subvert[""] = 0
   
        # Map cornac ID to a sequential index
        for key, value in news_ab_mapped.items():
            if key == -1:
                news_ab[key] = ""
            else:
                body_word = self.word_tokenize(value)
                news_ab[key] = body_word
          
        
        self.news_ab_index =  np.zeros((len(news_ab), self.body_size), dtype="int32")
        self.news_vert_index = np.zeros((len(news_vert), 1), dtype="int32")
        self.news_subvert_index = np.zeros((len(news_subvert), 1), dtype="int32")
        for key, ab in news_ab.items():
            mapped_index = self.news_index_map[key]
            for word_index in range(min(self.body_size, len(ab))):
                word = ab[word_index].lower()
                if word in self.word_dict:
                    self.news_ab_index[mapped_index,
                                          word_index] = self.word_dict[word]
                    
        for key, vert in news_vert.items():
            if key!= -1:
                mapped_index = self.news_index_map[key]
                self.news_vert_index[mapped_index, 0] = vert_dict[vert]
            
        for key, subvert in news_subvert.items():
            if key!=-1:
                mapped_index = self.news_index_map[key]
                self.news_subvert_index[mapped_index, 0] = subvert_dict[subvert]
        
            
        

    def process_history_news_ab(self, history_raw_IID, history_size):
        """init news information given news file, such as news_title_index.
        Args:
            news_file: path of news file
            history_raw_IID: raw item ids for a user
            history_size: the fixed history size to keep.
        """

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
            if i in self.news_ab:
                news_json.append(self.news_ab[i])
            elif i == -1:

                news_json.append("")

        news_text = []
        for value in news_json:

            body_text = self.word_tokenize(value)
            news_text.append(body_text)

        his_index = np.zeros(
            (len(news_text), self.body_size), dtype="int32"
        )
        # total news_title * word size
        for i in range(len(news_text)):
            body = news_text[i]
            for word_index in range(min(self.body_size, len(body))):
                word = body[word_index].lower()
                if word in self.word_dict:
                    his_index[i, word_index] = self.word_dict[word]
        return his_index
        

    def process_history_news_vert(self, history_raw_IID, history_size):
        """init news information given news file, such as news_title_index.
        Args:
            news_file: path of news file
            history_raw_IID: raw item ids for a user
            history_size: the fixed history size to keep.
        """
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

        his_index = np.array(
    [self.news_vert_index[self.news_index_map[key]] for key in history_raw_IID])
        # news_category_json = []
        # for i in history_raw_IID:
        #     if i in self.news_vert:
        #         vert = self.news_vert[i]
        #         news_category_json.append(self.vert_dict[vert])
     

        # his_index = np.zeros(
        #     (len(news_category_json), 1), dtype="int32"
        # )
        # # total news_title * word size
        # for i in range(len(news_category_json)):

        #     his_index[i, 0] = news_category_json[i]
        return his_index




    def process_history_news_subvert(self, history_raw_IID, history_size):
        """init news information given news file, such as news_title_index.
        Args:
            news_file: path of news file
            history_raw_IID: raw item ids for a user
            history_size: the fixed history size to keep.
        """

        def pad_or_truncate(sequence, max_length):
            if len(sequence) < max_length:
                # Pad with -1 if the sequence is too short
                return [-1] * (max_length - len(sequence)) + sequence
            else:
                # Truncate the sequence if it's too long
                return sequence[-max_length:]

        history_raw_IID = pad_or_truncate(history_raw_IID, history_size)
        his_index = np.array(
    [self.news_subvert_index[self.news_index_map[key]] for key in history_raw_IID])
        return his_index
        


        


   

        

            

