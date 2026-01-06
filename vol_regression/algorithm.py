import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialVolatilityPredictor:
    """
    一个使用多项式回归预测比特币波动率的类。
    """
    def __init__(self, degree=2):
        """
        初始化预测器。

        :param degree: 多项式回归的阶数。
        """
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()
        self.df = None

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
        # 计算波动率: (high - low) / close
        self.df['volatility'] = (self.df['high'] - self.df['low']) / self.df['close']
        
        # 计算方向: 如果收盘价 > 开盘价，则为1，否则为0
        self.df['direction'] = (self.df['close'] > self.df['open']).astype(int)
        
        # 创建一个时间索引作为回归特征
        self.df['time_idx'] = np.arange(len(self.df))

    def train(self):
        """
        训练多项式回归模型来预测波动率。
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data()。")
            
        X = self.df[['time_idx']]
        y = self.df['volatility']
        
        X_poly = self.poly_features.fit_transform(X)
        
        self.model.fit(X_poly, y)

    def predict(self, future_steps):
        """
        预测未来几天的波动率。

        :param future_steps: 要预测的未来天数。
        :return: 预测的波动率数组。
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()。")
            
        last_time_idx = self.df['time_idx'].max()
        future_time_idx = np.arange(last_time_idx + 1, last_time_idx + 1 + future_steps).reshape(-1, 1)
        
        future_time_idx_poly = self.poly_features.transform(future_time_idx)
        
        predicted_volatility = self.model.predict(future_time_idx_poly)
        
        return predicted_volatility

    def get_processed_data(self):
        """
        获取包含日期、计算出的波动率和方向的数据。

        :return: 一个包含 'date', 'volatility', 'direction' 列的DataFrame。
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data()。")
        return self.df[['date', 'volatility', 'direction']]
