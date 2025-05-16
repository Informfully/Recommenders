#############################################################################################################                              #
# DESCRIPTION:  Example for exposure diversity(epd) model experiment in an example dataset                        #
#############################################################################################################


import cornac
from cornac.datasets import mind
from cornac.models import EPD
from cornac.metrics import Recall


import random
import os
import json
import sys
from cornac.eval_methods import BaseMethod
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


news_files_dir = os.path.join(current_dir, 'example_news_files')

config_files_dir = os.path.join(current_dir, 'example_config_files')
sys.path.insert(0, news_files_dir)
sys.path.insert(0, config_files_dir)


input_path = news_files_dir

political_ref_path = os.path.join(input_path, 'political_reference.json') 
party_path = os.path.join(input_path, 'example_party.json') 


train_uir_path = os.path.join(input_path, 'example_impression_train_uir.csv')
feedback_train = mind.load_feedback(fpath = train_uir_path)

test_uir_path = os.path.join(input_path, 'example_impression_test_uir.csv')
feedback_test = mind.load_feedback(fpath = test_uir_path)

# article_pool_path = os.path.join(input_path, 'example_article_pool.csv')
# impression_items_df = pd.read_csv(article_pool_path, dtype ={'iid': str})
# impression_iid_list = impression_items_df['iid'].tolist()

rs = BaseMethod.from_splits(
    train_data=feedback_train,
    test_data=feedback_test,
    exclude_unknowns=False,
    verbose=True,
    rating_threshold=0.5
)



article_pool_path = os.path.join(input_path, 'example_article_pool.csv')
impression_items_df = pd.read_csv(article_pool_path, dtype ={'iid': str})
impression_iid_list = impression_items_df['iid'].tolist()

config_path = os.path.join(config_files_dir, 'model_parameters.ini')


political_type_dict = {1: 'neutral', 2:'major', 3:'minor'}
num_items = len(set([i for u,i,r in feedback_train]))

test_uir = list(zip(*rs.test_set.uir_tuple))



def load_user_group_type(uir):
    # Unique user IDs
    uid_set = set(uid for uid, _, _ in uir)
    uid_list = list(uid_set)
    
    # Shuffle for randomness
    random.shuffle(uid_list)
    
    total_users = len(uid_list)
    group_size = total_users // 3

    print(f"group_size average:{group_size}")
    
    # Split into 3 groups
    group1 = uid_list[:group_size]
    group2 = uid_list[group_size:2*group_size]
    group3 = uid_list[2*group_size:3*group_size]
    
    # Handle remainder users
    remainder = uid_list[3*group_size:]
    groups = {1: group1, 2: group2, 3: group3}
    choices = [1, 2, 3]
    
    for uid in remainder:
        group_choice = random.choice(choices)
        groups[group_choice].append(uid)
    print(f"groups len: {len(groups[1])}, {len(groups[2])}, {len(groups[3])}")
    
    # Build user_id to group_id mapping
    user_group_dict = {}
    for group_id, users in groups.items():
        for uid in users:
            user_group_dict[uid] = group_id
    
    return user_group_dict

user_group_dict = load_user_group_type(test_uir)
cleaned_dict = {int(k): v for k, v in user_group_dict.items()}

user_group_save_path = os.path.join(input_path, 'epd_user_groups.json')  # save user groups

with open(user_group_save_path, 'w') as f:
    json.dump(cleaned_dict, f, indent=4)



# initialize models
model_ebnerd = EPD(name = "EPD_ebnerd",
     party_path = party_path, political_type_dict= political_type_dict, num_items=num_items, configure_path = config_path,
        
        k=2, pageWidth = 20, article_pool = impression_iid_list, userGroupDict = user_group_dict, dataset_name = "ebnerd" , political_ref_path=political_ref_path)

model_mind = EPD(name = "EPD_mind",
                    party_path = party_path, political_type_dict= political_type_dict, num_items=num_items, configure_path = config_path,
        
        k=2, pageWidth = 20,  userGroupDict = user_group_dict, dataset_name = "mind" , political_ref_path=political_ref_path)


# define metrics to evaluate the models
metrics = [ Recall(k=20)]

# put it together in an experiment, voilà!
output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.', 'output_epd_result')
cornac.Experiment(
        eval_method=rs,
        models=[model_ebnerd, model_mind],
        metrics=metrics,
        save_dir=output_file_path
    ).run()