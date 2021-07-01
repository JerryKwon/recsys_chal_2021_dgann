import os
import re
from tqdm import tqdm
from collections import defaultdict

import pandas as pd


class LzoLoader:
    def __init__(self, data_path, verbose=False):
        self.verbose = verbose

        self.data_path = data_path
        self.unzipped_p = re.compile("part-[0-9]{5}$")

        self.all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links",
                             "present_domains", "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id",
                             "engaged_with_user_follower_count", "engaged_with_user_following_count", "engaged_with_user_is_verified",
                             "engaged_with_user_account_creation", "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                             "engaging_user_is_verified", "engaging_user_account_creation", "engagee_follows_engager"]

        self.all_features_to_idx = dict(zip(self.all_features, range(len(self.all_features))))
        self.labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22,
                              "like_timestamp": 23}

    def load_data(self, list_lzo):
        list_data = os.listdir(self.data_path)
        list_unzipped = [file for file in list_data if self.unzipped_p.match(file)]
        target_lzos = ["part-" + format(lzo, "05d") for lzo in list_lzo]

        lzo_diff = set(target_lzos).difference(set(list_unzipped))

        if len(lzo_diff) > 0:
            raise Exception(f"{lzo_diff} are not in data_path")

        if self.verbose:
            print(f"import lzo {target_lzos}; # of lzo: {len(target_lzos)}")

        all_dictionary = defaultdict(list)

        for file in (tqdm(target_lzos, desc="importing dataset") if self.verbose else target_lzos):
            with open(os.path.join(self.data_path, file), encoding="utf-8") as f:
                if self.verbose:
                    print(f"\nimporting {file}" + "...")
                for index, line in enumerate(f.readlines()):
                    line = line.strip()
                    features = line.split("\x01")
                    for feature, idx in self.all_features_to_idx.items():
                        feat = features[idx]
                        if feature == "text_ tokens":
                            feat = features[idx].replace("\t", " ")
                        all_dictionary[feature].append(feat)

                    for label, idx in self.labels_to_idx.items():
                        lab = features[idx]
                        all_dictionary[label].append(lab)

        return pd.DataFrame(all_dictionary)
