# This file is used to find the overlapping time range between two datasets
import pandas as pd
import os

# data paths
BTC_PATH = os.path.join("data", "Bitcoin_history_data.csv")
TWEETS_PATH = os.path.join("data", "tweets.csv")

# Load data
btc = pd.read_csv(BTC_PATH)
tweets = pd.read_csv(TWEETS_PATH)

# Parse datetime
tweets["date"] = pd.to_datetime(tweets["date"], errors="coerce")
btc["Date"] = pd.to_datetime(btc["Date"], errors="coerce")

# Normalize to daily (BTC is daily data)
tweets["day"] = tweets["date"].dt.normalize()
btc["day"] = btc["Date"].dt.normalize()

# Split tweets by year
tweet_train = tweets[tweets["date"].dt.year == 2022].reset_index(drop=True)
tweet_test  = tweets[tweets["date"].dt.year == 2023].reset_index(drop=True)

# Split BTC by tweet days (use tweet time as reference)
train_days = tweet_train["day"].unique()
test_days  = tweet_test["day"].unique()

btc_train = btc[btc["day"].isin(train_days)].reset_index(drop=True)
btc_test  = btc[btc["day"].isin(test_days)].reset_index(drop=True)

# Save to current directory
btc_train.to_csv("./data/btc_train.csv", index=False)
btc_test.to_csv("./data/btc_test.csv", index=False)
tweet_train.to_csv("./data/tweet_train.csv", index=False)
tweet_test.to_csv("./data/tweet_test.csv", index=False)


# Sanity check
print("Saved to current directory:")
print(f"Tweet Train (2022): {len(tweet_train)} rows,"
      f" {tweet_train['day'].nunique()} days")
print(f"Tweet Test  (2023): {len(tweet_test)} rows,"
      f" {tweet_test['day'].nunique()} days")
print(f"BTC Train: {len(btc_train)} rows,"
      f" {btc_train['day'].nunique()} days")
print(f"BTC Test : {len(btc_test)} rows,"
      f" {btc_test['day'].nunique()} days")