import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
        
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        # 计算波动率: (high - low) / close
        self.df['volatility'] = (self.df['high'] - self.df['low']) / self.df['close']
        
        # 计算方向: 如果收盘价 > 开盘价，则为1，否则为0
        self.df['direction'] = (self.df['close'] > self.df['open']).astype(int)
        
        # Make sure data is ordered by time if a date column exists
        if "date" in self.df.columns:
            try:
                self.df["date"] = pd.to_datetime(self.df["date"])
                self.df = self.df.sort_values("date").reset_index(drop=True)
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

        required = {"high", "low", "close"}
        missing = required - set(dfw.columns)
        if missing:
            raise ValueError(f"predict() input missing columns: {sorted(missing)}")

        if len(dfw) != self.window_size:
            raise ValueError(
                f"Expected recent_ohlc window length {self.window_size}, got {len(dfw)}"
            )

        high = dfw["high"].to_numpy(dtype=float)
        low = dfw["low"].to_numpy(dtype=float)
        close = dfw["close"].to_numpy(dtype=float)

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
        return self.df[['date', 'volatility', 'direction']]
