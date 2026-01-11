import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor


class PolynomialVolatilityPredictor:
    """
    一个使用多项式回归预测比特币波动率的类。
    """
    def __init__(self, degree: int=2, 
                 window_size: int=30, 
                 forecast_window: int=7):
        """
        初始化预测器。

        :param degree: 多项式回归的阶数。
        :param window_size: 过去的时间窗
        :param forecast_window: 预测的时间窗。
        """

        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if forecast_window < 1:
            raise ValueError("forecast_window must be at least 1.")
        if degree < 1:
            raise ValueError("degree must be at least 1.")
        self.degree = degree
        self.window_size = window_size
        self.forecast_window = forecast_window
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.model = LinearRegression()
        self.df = None
        self._is_trained = False

    def load_data(self, filepath):
        """
        从CSV文件加载数据并进行预处理。

        :param filepath: 数据文件的路径。
        """
        self.df = pd.read_csv(filepath)
        self._preprocess_data()

    def _preprocess_data(self):
        """
        预处理数据，计算波动率和方向。
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please load data before preprocessing.")
        
        required_cols = {"Open", "High", "Low", "Close"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        # 计算波动率: (high - low) / close
        self.df['volatility'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        
        # 计算方向: 如果收盘价 > 开盘价，则为1，否则为0
        self.df['direction'] = (self.df['Close'] > self.df['Open']).astype(int)
        
        # Make sure data is ordered by time if a date column exists
        if "Date" in self.df.columns:
            try:
                self.df["Date"] = pd.to_datetime(self.df["Date"])
                self.df = self.df.sort_values("Date").reset_index(drop=True)
            except Exception:
                self.df = self.df.reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

    def _build_supervised_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, Y) for multi-step forecasting.

        Returns:
            X: shape (n_samples, window_size)
            Y: shape (n_samples, forecast_size)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        vol = self.df["volatility"].to_numpy(dtype=float)
        W = self.window_size
        H = self.forecast_window

        n_total = len(vol)
        n_samples = n_total - W - H + 1
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data to build samples: need at least window_size({W}) + "
                f"forecast_size({H}) points, but got {n_total}."
            )

        X = np.zeros((n_samples, W), dtype=float)
        Y = np.zeros((n_samples, H), dtype=float)

        # For each sample i, predict horizon starting at t = i + W
        for i in range(n_samples):
            X[i, :] = vol[i : i + W]
            Y[i, :] = vol[i + W : i + W + H]

        return X, Y

    def train(self):
        """
        训练多项式回归模型来预测波动率。
        """
        X, Y = self._build_supervised_xy()
        X_poly = self.poly_features.fit_transform(X)
        self.model.fit(X_poly, Y)
        self._is_trained = True


    def _compute_volatility_from_ohlc_window(self, recent_ohlc) -> np.ndarray:
        if isinstance(recent_ohlc, pd.DataFrame):
            dfw = recent_ohlc.copy()
        else:
            # assume list of dicts / records
            dfw = pd.DataFrame(recent_ohlc)

        required = {"High", "Low", "Close"}
        missing = required - set(dfw.columns)
        if missing:
            raise ValueError(f"predict() input missing columns: {sorted(missing)}")

        if len(dfw) != self.window_size:
            raise ValueError(
                f"Expected recent_ohlc window length {self.window_size}, got {len(dfw)}"
            )

        high = dfw["High"].to_numpy(dtype=float)
        low = dfw["Low"].to_numpy(dtype=float)
        close = dfw["Close"].to_numpy(dtype=float)

        # avoid division-by-zero
        if np.any(close == 0):
            raise ValueError("Found close=0 in the input window; cannot compute volatility.")

        vol_vec = (high - low) / close  # shape (window_size,)
        return vol_vec

    def predict(self, recent_ohlc:np.ndarray) -> np.ndarray:
        """
        使用训练好的模型预测未来的波动率。

        :param recent_ohlc: 最近的OHLC数据。
        :return: 预测的未来波动率，形状为 (forecast_window,)。
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Please call train() before predict().")

        vol_vec = self._compute_volatility_from_ohlc_window(recent_ohlc)
        X = vol_vec.reshape(1, -1)  # shape (1, window_size)
        X_poly = self.poly_features.transform(X) 
        y_pred = self.model.predict(X_poly)  # shape (1, forecast_window)
        return y_pred.reshape(-1) # shape (forecast_window,)

        
        

    def get_processed_data(self):
        """
        获取包含日期、计算出的波动率和方向的数据。

        :return: 一个包含 'date', 'volatility', 'direction' 列的DataFrame。
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data()。")
        return self.df[['Date', 'volatility', 'direction']]


# class PolynomialMultiModalVolatilityPredictor:
#     def __init__(self, degree=4, window_size=30, forecast_window=7):
#         self.degree = int(degree)
#         self.window_size = int(window_size)
#         self.forecast_window = int(forecast_window)

#         self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
#         self.model = LinearRegression()

#         self.df_daily = None
#         self._trained = False

#         self.tweet_cols = [
#             "sentiment_mean_t",
#             "sentiment_std_t",
#             "tweet_count_t",
#             "positive_ratio_t",
#             "negative_ratio_t",
#         ]

#         # PCA
#         self.pca_components = 319 # 超参，可调整
#         self.pca = PCA(n_components=self.pca_components)


#     def load_daily_table(self, df_daily: pd.DataFrame):
#         required = {"Date", "volatility", *self.tweet_cols}
#         missing = required - set(df_daily.columns)
#         if missing:
#             raise ValueError(f"daily table missing columns: {sorted(missing)}")

#         df = df_daily.copy()
#         df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#         df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

#         # ensure numeric
#         df["volatility"] = pd.to_numeric(df["volatility"], errors="coerce")
#         df = df.dropna(subset=["volatility"])
#         for c in self.tweet_cols:
#             df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

#         self.df_daily = df

#     def _build_xy(self):
#         W, H = self.window_size, self.forecast_window
#         vol = self.df_daily["volatility"].to_numpy(float)                 # (T,)
#         tw  = self.df_daily[self.tweet_cols].to_numpy(float)              # (T,5)

#         T = len(vol)
#         n = T - W - H + 1
#         if n <= 0:
#             raise ValueError(f"Not enough data: need at least W+H={W+H}, got {T}")

#         X = np.zeros((n, W * 6), dtype=float)  # 6 dims/day
#         Y = np.zeros((n, H), dtype=float)

#         for i in range(n):
#             vol_win = vol[i:i+W].reshape(W, 1)      # (W,1)
#             tw_win  = tw[i:i+W]                     # (W,5)
#             day_win = np.concatenate([vol_win, tw_win], axis=1)  # (W,6)
#             X[i] = day_win.reshape(-1)
#             Y[i] = vol[i+W:i+W+H]

#         return X, Y

#     def train(self):
#         X, Y = self._build_xy()
#         Xp = self.poly.fit_transform(X)
#         # PCA 
#         Xp = self.pca.fit_transform(Xp)
#         self.model.fit(Xp, Y) # multi-output
#         self._trained = True

#     def predict(self, recent_daily_window: pd.DataFrame) -> np.ndarray:
#         if not self._trained:
#             raise ValueError("Model not trained. Call train() first.")
#         if len(recent_daily_window) != self.window_size:
#             raise ValueError(f"Expected window length {self.window_size}, got {len(recent_daily_window)}")

#         cols = ["volatility"] + self.tweet_cols
#         missing = set(cols) - set(recent_daily_window.columns)
#         if missing:
#             raise ValueError(f"predict window missing columns: {sorted(missing)}")

#         x = recent_daily_window[cols].to_numpy(float).reshape(1, -1)  # (1, W*6)
#         xp = self.poly.transform(x)
#         #PCA
#         xp = self.pca.transform(xp)
#         y = self.model.predict(xp)  # (1, H)
#         return y.reshape(-1)


# catboost 
class PolynomialMultiModalVolatilityPredictor:
    def __init__(self, degree=4, window_size=30, forecast_window=7):
        self.degree = int(degree)
        self.window_size = int(window_size)
        self.forecast_window = int(forecast_window)

        # catboost_params = {
        #     'iterations': 300,
        #     'learning_rate': 0.1,
        #     'depth': 6,
        #     'verbose': 0,  # 训练时不打印日志
        #     'l2_leaf_reg': 1,
        #     'bootstrap_type': 'Bernoulli',
        #     'subsample': 0.8,
        #     'loss_function': 'MultiRMSE',   # 使用CatBoost原生的多目标损失函数
        #     'eval_metric': 'MultiRMSE',
        #     'early_stopping_rounds': 50     # 如果验证集性能50轮内无提升，则停止训练
        # }

        catboost_params = {
            'iterations': 200,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False
        }
        self.model = MultiOutputRegressor(CatBoostRegressor(**catboost_params))


        self.df_daily = None
        self._trained = False

        self.tweet_cols = [
            "sentiment_mean_t",
            "sentiment_std_t",
            "tweet_count_t",
            "positive_ratio_t",
            "negative_ratio_t",
        ]

        # PCA
        self.pca_components = 319 # 超参，可调整
        self.pca = PCA(n_components=self.pca_components)


    def load_daily_table(self, df_daily: pd.DataFrame):
        required = {"Date", "volatility", *self.tweet_cols}
        missing = required - set(df_daily.columns)
        if missing:
            raise ValueError(f"daily table missing columns: {sorted(missing)}")

        df = df_daily.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # ensure numeric
        df["volatility"] = pd.to_numeric(df["volatility"], errors="coerce")
        df = df.dropna(subset=["volatility"])
        for c in self.tweet_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        self.df_daily = df

    def _build_xy(self):
        W, H = self.window_size, self.forecast_window
        vol = self.df_daily["volatility"].to_numpy(float)                 # (T,)
        tw  = self.df_daily[self.tweet_cols].to_numpy(float)              # (T,5)

        T = len(vol)
        n = T - W - H + 1
        if n <= 0:
            raise ValueError(f"Not enough data: need at least W+H={W+H}, got {T}")

        X = np.zeros((n, W * 6), dtype=float)  # 6 dims/day
        Y = np.zeros((n, H), dtype=float)

        for i in range(n):
            vol_win = vol[i:i+W].reshape(W, 1)      # (W,1)
            tw_win  = tw[i:i+W]                     # (W,5)
            day_win = np.concatenate([vol_win, tw_win], axis=1)  # (W,6)
            X[i] = day_win.reshape(-1)
            Y[i] = vol[i+W:i+W+H]

        return X, Y

    def train(self):
        X, Y = self._build_xy()
        # Xp = self.poly.fit_transform(X)
        # # PCA 
        # Xp = self.pca.fit_transform(Xp)
        # self.model.fit(Xp, Y) # multi-output
        self.model.fit(X, Y)
        self._trained = True

    def predict(self, recent_daily_window: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise ValueError("Model not trained. Call train() first.")
        if len(recent_daily_window) != self.window_size:
            raise ValueError(f"Expected window length {self.window_size}, got {len(recent_daily_window)}")

        cols = ["volatility"] + self.tweet_cols
        missing = set(cols) - set(recent_daily_window.columns)
        if missing:
            raise ValueError(f"predict window missing columns: {sorted(missing)}")

        x = recent_daily_window[cols].to_numpy(float).reshape(1, -1)  # (1, W*6)
        # xp = self.poly.transform(x)
        # #PCA
        # xp = self.pca.transform(xp)
        y = self.model.predict(x)  # (1, H)
        return y.reshape(-1)

