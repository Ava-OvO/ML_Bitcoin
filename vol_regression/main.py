from algorithm import PolynomialVolatilityPredictor
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def baseline_pred(window_size, forecast_window, predictor, test_data_path = None):
    if test_data_path:
        test_df = pd.read_csv(test_data_path)
        if "data" in test_df.columns:
            test_df['data'] = pd.to_datetime(test_df['data'])
            test_df = test_df.sort_values("data").reset_index(drop=True)
        else:
            test_df = test_df.reset_index(drop=True)

        needed_cols = {"high", "low", "close"}
        if not needed_cols.issubset(set(test_df.columns)):
            raise ValueError(f"Test data must contain columns: {needed_cols}")
        
        if len(test_df) < (window_size + forecast_window):
            raise ValueError(f"Test data must have at least {window_size + forecast_window} rows.")
        
        future_true_window = test_df.iloc[window_size:window_size + forecast_window][["high", "low", "close"]]
        true_vol = (future_true_window["high"] - future_true_window["low"]) / future_true_window["close"]
        true_vol = true_vol.to_numpy(dtype=float)

        Input = test_df.iloc[0:window_size][["high", "low", "close"]]
        future_pred = predictor.predict(Input)
        
        err = future_pred - true_vol
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

        print("\nGround-truth volatility (from test set first forecast window):")
        print(true_vol)
        print(f"\nMAE={mae:.6f}, RMSE={rmse:.6f}")

        x = np.arange(1, forecast_window + 1)
        plt.figure()
        plt.plot(x, true_vol, label = "true")
        plt.plot(x, future_pred, label = "pred")
        plt.xlabel("Days into the future")
        plt.ylabel("Volatility")
        plt.title("Volatility Prediction vs True Values")
        plt.legend()
        plt.tight_layout()
        plt.savefig("volatility_prediction_vs_true.png")
        plt.show()


def baseline(poly_degree=4, 
             train_data_path = None,
             test_data_path = None,
             window_size = 30,
             forcast_window = 7
             ):
    # print("test")
     # 1. 初始化预测器，可以指定多项式的阶数
    predictor = PolynomialVolatilityPredictor(degree=poly_degree, window_size=window_size, forecast_window=forcast_window)
    
    # 2. 加载数据
    # 注意：请确保文件路径正确
    try:
        if not train_data_path:
            raise ValueError("Data path must be provided.")
        predictor.load_data(train_data_path)

        # 3. 训练模型
        predictor.train()
        
        # 4. 获取处理后的数据（包含volatility和direction）
        processed_data = predictor.get_processed_data()
        # print("处理后的数据 (前5行):")
        print("Processed data (first 5 rows):")
        print(processed_data.head())
        
        # 5. 预测未来的波动率
        baseline_pred(window_size, forcast_window, predictor, test_data_path=test_data_path)

    except FileNotFoundError:
        # print("错误: toy_btc_ohlcv.csv 文件未找到。请检查路径是否正确。")
        print("Error: toy_btc_ohlcv.csv file not found. Please check if the path is correct.")
    except Exception as e:
        # print(f"发生错误: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polynomial Volatility Predictor")
    parser.add_argument("--poly_degree", type=int, default=4, help="Degree of the polynomial")
    parser.add_argument("--train_data_path", type=str, default=r"data\toy_btc_ohlcv.csv", help="Path to the training data file")
    parser.add_argument("--test_data_path", type=str, default=r"data\toy_btc_ohlcv.csv", help="Path to the testing data file")
    parser.add_argument("--model", type = str, default = "baseline", help = "Model to use: baseline or advanced")
    parser.add_argument("--window_size", type = int, default = 30, help = "Window size for training")
    parser.add_argument("--forcast_window", type = int, default = 7, help = "Forecast window size")

    args = parser.parse_args()

    if args.model == "baseline":
        baseline(poly_degree=args.poly_degree,
             train_data_path=args.train_data_path,
             test_data_path=args.test_data_path,
             window_size=args.window_size,
             forcast_window=args.forcast_window)
    elif args.model == "advanced":
        # todo
        pass
    else:
        print(f"Unknown model: {args.model}")