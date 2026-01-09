import pandas as pd
import numpy as np

def build_daily_tweet_features(
    tweets_df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "sentiment_score",
    pos_thr: float = 0.0,
    neg_thr: float = 0.0,
) -> pd.DataFrame:
    """
    Return a daily dataframe with columns:
      day, sentiment_mean_t, sentiment_std_t, tweet_count_t, positive_ratio_t, negative_ratio_t
    """
    df = tweets_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["Date"] = df[date_col].dt.normalize()

    if score_col not in df.columns:
        raise ValueError(
            f"tweets_df must contain '{score_col}'. "
            f"If you don't have it yet, you need to compute sentiment_score first."
        )

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    def agg_day(g: pd.DataFrame) -> pd.Series:
        s = g[score_col].dropna()
        n = len(g)  # tweet_count includes even NaN scores; change to len(s) if you prefer scored-only
        if len(s) == 0:
            mean = np.nan
            std = np.nan
            pos_ratio = np.nan
            neg_ratio = np.nan
        else:
            mean = float(s.mean())
            std = float(s.std(ddof=0))  # population std
            pos_ratio = float((s > pos_thr).mean())
            neg_ratio = float((s < neg_thr).mean())

        return pd.Series({
            "sentiment_mean_t": mean,
            "sentiment_std_t": std,
            "tweet_count_t": int(n),
            "positive_ratio_t": pos_ratio,
            "negative_ratio_t": neg_ratio,
        })

    daily = df.groupby("Date", sort=True).apply(agg_day, include_groups = False).reset_index()
    return daily

def build_daily_btc_vol(btc_df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = btc_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df["Date"] = df[date_col].dt.normalize()

    # volatility: (high-low)/close
    for c in ["High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["High", "Low", "Close"])
    df["volatility"] = (df["High"] - df["Low"]) / df["Close"]
    return df[["Date", "volatility"]].drop_duplicates("Date")

def build_daily_multimodal_table(btc_daily: pd.DataFrame, tweet_daily: pd.DataFrame) -> pd.DataFrame:
    df = btc_daily.merge(tweet_daily, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    # fill NaN values. Now we use 0, but the statistical values (mean, std) may be better.
    fill_cols = ["sentiment_mean_t", "sentiment_std_t", "positive_ratio_t", "negative_ratio_t"]
    df[fill_cols] = df[fill_cols].fillna(0.0)
    df["tweet_count_t"] = df["tweet_count_t"].fillna(0).astype(int)

    return df
