import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

data = pd.read_csv("movielens_preprocessed_data.csv")
print(data.info())
# print(data.columns)

print("\n===== DATA — REWARD PROPORTIONS =====")
print(data['reward'].value_counts(normalize=True))


print("\n===== DATA — NUMBER OF UNIQUE USERS =====")
print(data['user_id'].nunique())

print("\n===== DATA — NUMBER OF UNIQUE ITEMS =====")
print(data['item_id'].nunique())

print("\n===== DATA — USER ACTIVITY (INTERACTIONS PER USER) =====")
print(data['user_id'].value_counts().describe())

print("\n===== DATA — USER REWARD RATE DISTRIBUTION =====")
user_reward = data.groupby('user_id')['reward'].mean()
print(user_reward.describe())


print("\n===== DATA — ITEM POPULARITY (INTERACTIONS PER ITEM) =====")
print(data['item_id'].value_counts().describe())

print("\n===== DATA — ITEM REWARD RATE DISTRIBUTION =====")
item_reward = data.groupby('item_id')['reward'].mean()
print(item_reward.describe())

data['desc_len'] = data['description'].str.len()

print("\n===== DATA — DESCRIPTION LENGTH STATS =====")
print(data['desc_len'].describe())

print("\n===== DATA — AVERAGE DESCRIPTION LENGTH BY REWARD =====")
print(data.groupby('reward')['desc_len'].mean())


words = data['description'].str.lower().str.cat(sep=' ')
words = re.findall(r'\\b[a-z]{3,}\\b', words)
common_words = Counter(words).most_common(20)

print("\n===== DATA — TOP 20 MOST COMMON WORDS IN DESCRIPTIONS =====")
print(common_words)

print("\n===== DATA — TOP 10 MOVIES BY REWARD RATE =====")
print(data.groupby('item_id')['reward'].mean().sort_values(ascending=False).head(10))

print("\n===== DATA — FIRST 10 USERS BY REWARD RATE =====")
print(data.groupby('user_id')['reward'].mean().head(10))


data_sample = data.sample(frac=0.1, random_state=42)
print("\n===== DATA — SAMPLE SHAPE (10% REPRODUCIBLE SAMPLE) =====")
print(data_sample.shape)


######### 15% Data Sample ###############
sample_frac = 0.15
data_sample = data.sample(frac=sample_frac, random_state=42)
data_sample.to_csv(
    "movielens_preprocessed_data_sample_15.csv",
    index=False
)