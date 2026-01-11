from algorithm import PolynomialVolatilityPredictor, PolynomialMultiModalVolatilityPredictor
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from advance_input import build_daily_btc_vol, build_daily_tweet_features, build_daily_multimodal_table
from tqdm import tqdm

# def baseline_pred(window_size, forecast_window, predictor, test_data_path = None):
#     if test_data_path:
#         test_df = pd.read_csv(test_data_path)
#         if "Date" in test_df.columns:
#             test_df['Date'] = pd.to_datetime(test_df['Date'])
#             test_df = test_df.sort_values("Date").reset_index(drop=True)
#         else:
#             test_df = test_df.reset_index(drop=True)

#         needed_cols = {"High", "Low", "Close"}
#         if not needed_cols.issubset(set(test_df.columns)):
#             raise ValueError(f"Test data must contain columns: {needed_cols}")
        
#         if len(test_df) < (window_size + forecast_window):
#             raise ValueError(f"Test data must have at least {window_size + forecast_window} rows.")
        
#         future_true_window = test_df.iloc[window_size:window_size + forecast_window][["High", "Low", "Close"]]
#         true_vol = (future_true_window["High"] - future_true_window["Low"]) / future_true_window["Close"]
#         true_vol = true_vol.to_numpy(dtype=float)

#         Input = test_df.iloc[0:window_size][["High", "Low", "Close"]]
#         future_pred = predictor.predict(Input)
        
#         err = future_pred - true_vol
#         mae = float(np.mean(np.abs(err)))
#         rmse = float(np.sqrt(np.mean(err ** 2)))

#         print("Baseline Model Prediction Results:")
#         print("Pred:", future_pred)
#         print("True:", true_vol)
#         print(f"\nMAE={mae:.6f}, RMSE={rmse:.6f}")

#         x = np.arange(1, forecast_window + 1)
#         plt.figure()
#         plt.plot(x, true_vol, label = "true")
#         plt.plot(x, future_pred, label = "pred")
#         plt.xlabel("Days into the future")
#         plt.ylabel("Volatility")
#         plt.title("Volatility Prediction vs True Values")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig("./vol_regression/baseline_volatility_prediction_vs_true.png")
#         # plt.show()

#         print("Baseline true days:", test_df.iloc[window_size:window_size+forecast_window]["Date"].tolist())

def baseline_pred(window_size, forecast_window, predictor, test_data_path=None):
    if not test_data_path:
        raise ValueError("test_data_path must be provided for baseline_pred.")

    test_df = pd.read_csv(test_data_path)

    # sort by date if possible
    if "Date" in test_df.columns:
        test_df["Date"] = pd.to_datetime(test_df["Date"])
        test_df = test_df.sort_values("Date").reset_index(drop=True)
        has_date = True
    else:
        test_df = test_df.reset_index(drop=True)
        has_date = False

    needed_cols = {"High", "Low", "Close"}
    if not needed_cols.issubset(set(test_df.columns)):
        raise ValueError(f"Test data must contain columns: {needed_cols}")

    if len(test_df) < (window_size + forecast_window):
        raise ValueError(f"Test data must have at least {window_size + forecast_window} rows.")

    maes = []
    rmses = []
    predictions_by_key = {}  # date -> [preds] (or index key if no Date)

    num_windows = len(test_df) - window_size - forecast_window + 1
    if num_windows <= 0:
        print("Not enough data for the given window size and forecast window.")
        return

    print(f"Evaluating Baseline Model over {num_windows} windows...")

    for i in tqdm(range(num_windows), desc="Evaluating Baseline Model"):
        input_window = test_df.iloc[i : i + window_size][["High", "Low", "Close"]]
        target_window = test_df.iloc[i + window_size : i + window_size + forecast_window][["High", "Low", "Close"]]

        # true volatility for the forecast window
        true_vol = ((target_window["High"] - target_window["Low"]) / target_window["Close"]).to_numpy(dtype=float)

        # model prediction
        pred = predictor.predict(input_window)

        err = pred - true_vol
        maes.append(float(np.mean(np.abs(err))))
        rmses.append(float(np.sqrt(np.mean(err ** 2))))

        # collect predictions for averaging (by Date if exists, else by absolute index)
        if has_date:
            target_keys = test_df.iloc[i + window_size : i + window_size + forecast_window]["Date"].tolist()
        else:
            # unique key per target point
            target_keys = list(range(i + window_size, i + window_size + forecast_window))

        for j, k in enumerate(target_keys):
            predictions_by_key.setdefault(k, []).append(float(pred[j]))

    # averaged predictions
    averaged_predictions = {k: float(np.mean(v)) for k, v in predictions_by_key.items()}

    # build aligned pred/true series for plotting
    if has_date:
        pred_series = pd.Series(averaged_predictions).sort_index()
        # true series: compute daily vol for all rows, then pick pred dates
        full_true_vol = ((test_df["High"] - test_df["Low"]) / test_df["Close"]).to_numpy(dtype=float)
        true_series = pd.Series(full_true_vol, index=test_df["Date"]).loc[pred_series.index]
        x_axis = pred_series.index
        x_label = "Date"
    else:
        pred_series = pd.Series(averaged_predictions).sort_index()
        full_true_vol = ((test_df["High"] - test_df["Low"]) / test_df["Close"]).to_numpy(dtype=float)
        true_series = pd.Series(full_true_vol, index=np.arange(len(test_df))).loc[pred_series.index]
        x_axis = pred_series.index
        x_label = "Index"

    avg_mae = float(np.mean(maes))
    avg_rmse = float(np.mean(rmses))

    print("\nBaseline Model Sliding Window Evaluation Results:")
    print(f"Average MAE across all windows: {avg_mae:.6f}")
    print(f"Average RMSE across all windows: {avg_rmse:.6f}")

    # plot
    plt.figure(figsize=(15, 7))
    plt.plot(x_axis, true_series.values, label="True Volatility", marker="o", linestyle="-")
    plt.plot(x_axis, pred_series.values, label="Averaged Prediction", marker="x", linestyle="--")
    plt.xlabel(x_label)
    plt.ylabel("Volatility")
    plt.title("Baseline Model: Averaged Volatility Prediction vs. Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./vol_regression/baseline_volatility_prediction_vs_true.png")

    print("\nPlot saved to ./vol_regression/baseline_volatility_prediction_vs_true.png")



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
        # print("Processed data (first 5 rows):")
        # print(processed_data.head())
        
        # 5. 预测未来的波动率
        baseline_pred(window_size, forcast_window, predictor, test_data_path=test_data_path)

    except FileNotFoundError:
        # print("错误: toy_btc_ohlcv.csv 文件未找到。请检查路径是否正确。")
        print(f"Error: {train_data_path} file not found. Please check if the path is correct.")
    # except Exception as e:
    #     # print(f"发生错误: {e}")
    #     print(f"An error occurred: {e}")

# def advanced_pred(window_size, forecast_window, model, daily_test):
#     if len(daily_test) < (window_size + forecast_window):
#         raise ValueError(f"Test data must have at least {window_size + forecast_window} rows.")
    
#     recent_window = daily_test.iloc[:window_size]
#     pred = model.predict(recent_window)

#     true_future = daily_test.iloc[window_size:window_size + forecast_window]["volatility"].to_numpy(dtype=float)
#     err = pred - true_future
#     mae = float(np.mean(np.abs(err)))
#     rmse = float(np.sqrt(np.mean(err ** 2)))

#     print("Advanced Model Prediction Results:")
#     print("Pred:", pred)
#     print("True:", true_future)
#     print(f"MAE={mae:.6f}, RMSE={rmse:.6f}")

#     # plot
#     x = np.arange(1, forecast_window + 1)
#     plt.figure()
#     plt.plot(x, true_future, label = "true", marker='o')
#     plt.plot(x, pred, label = "pred", marker='x')
#     plt.xlabel("Days into the future")
#     plt.ylabel("Volatility")
#     plt.title("Advanced Poly: Volatility Prediction vs Ground Truth")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("./vol_regression/advanced_poly_vol_prediction_vs_true.png")
#     # plt.show()

#     print("Advanced true days:", daily_test.iloc[window_size:window_size+forecast_window]["Date"].tolist())

def advanced_pred(window_size, forecast_window, model, daily_test):
    if len(daily_test) < (window_size + forecast_window):
        raise ValueError(f"Test data must have at least {window_size + forecast_window} rows.")
    
    maes = []
    rmses = []
    predictions_by_date = {}

    num_windows = len(daily_test) - window_size - forecast_window + 1
    if num_windows <= 0:
        print("Not enough data for the given window size and forecast window.")
        return
    
    print(f"Evaluating over {num_windows} windows...")

    for i in tqdm(range(num_windows), desc="Evaluating Advanced Model"):
        input_window = daily_test.iloc[i : i + window_size]
        target_window = daily_test.iloc[i + window_size : i + window_size + forecast_window]

        pred = model.predict(input_window)
        
        true_vol = target_window["volatility"].to_numpy(dtype=float)
        target_dates = target_window["Date"]

        err = pred - true_vol
        maes.append(np.mean(np.abs(err)))
        rmses.append(np.sqrt(np.mean(err ** 2)))

        for j, date in enumerate(target_dates):
            if date not in predictions_by_date:
                predictions_by_date[date] = []
            predictions_by_date[date].append(pred[j])

    averaged_predictions = {date: np.mean(preds) for date, preds in predictions_by_date.items()}
    
    pred_series = pd.Series(averaged_predictions).sort_index()
    
    true_series = daily_test.set_index("Date")["volatility"].loc[pred_series.index]

    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    
    print("\nAdvanced Model Sliding Window Evaluation Results:")
    print(f"Average MAE across all windows: {avg_mae:.6f}")
    print(f"Average RMSE across all windows: {avg_rmse:.6f}")

    plt.figure(figsize=(15, 7))
    plt.plot(true_series.index, true_series.values, label="True Volatility", marker='o', linestyle='-')
    plt.plot(pred_series.index, pred_series.values, label="Averaged Prediction", marker='x', linestyle='--')
    
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("Advanced Model: Averaged Volatility Prediction vs. Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./vol_regression/advanced_poly_vol_prediction_vs_true.png")

    print("\nPlot saved to ./vol_regression/advanced_poly_vol_prediction_vs_true.png")


def advanced(poly_degree=4, 
             train_btc_data_path = None,
             test_btc_data_path = None,
             train_tweet_data_path = None,
             test_tweet_data_path = None,
             window_size = 30,
             forcast_window = 7
             ):
    # load data
    btc_trian = pd.read_csv(train_btc_data_path)
    btc_test = pd.read_csv(test_btc_data_path)
    tweet_train = pd.read_csv(train_tweet_data_path)
    tweet_test = pd.read_csv(test_tweet_data_path)

    # build daily features
    btc_daily_train = build_daily_btc_vol(btc_trian, date_col="Date")
    btc_daily_test = build_daily_btc_vol(btc_test, date_col="Date")
    tw_daily_train = build_daily_tweet_features(tweet_train, date_col="date", score_col="sentiment_score")
    tw_daily_test = build_daily_tweet_features(tweet_test, date_col="date", score_col="sentiment_score")
    daily_train = build_daily_multimodal_table(btc_daily_train, tw_daily_train)
    daily_test = build_daily_multimodal_table(btc_daily_test, tw_daily_test)

    # train the model
    model = PolynomialMultiModalVolatilityPredictor(
        degree=poly_degree,
        window_size=window_size,
        forecast_window=forcast_window,
    )
    model.load_daily_table(daily_train)
    model.train()

    print("hello")
    # evaluate
    advanced_pred(window_size, forcast_window, model, daily_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polynomial Volatility Predictor")
    parser.add_argument("--poly_degree", type=int, default=3, help="Degree of the polynomial")
    parser.add_argument("--train_btc_data_path", type=str, default=r"data\btc_train.csv", help="Path to the training data file")
    parser.add_argument("--test_btc_data_path", type=str, default=r"data\btc_test.csv", help="Path to the testing data file")
    parser.add_argument("--train_tweet_data_path", type=str, default=r"data\tweet_train.csv", help="Path to the training tweet data file")
    parser.add_argument("--test_tweet_data_path", type=str, default=r"data\tweet_test.csv", help="Path to the testing tweet data file")
    parser.add_argument("--model", type = str, default = "advanced", help = "Model to use: baseline or advanced")
    parser.add_argument("--window_size", type = int, default = 50, help = "Window size for training")
    parser.add_argument("--forcast_window", type = int, default = 1, help = "Forecast window size")

    args = parser.parse_args()

    if args.model == "baseline":
        baseline(poly_degree=args.poly_degree,
             train_data_path=args.train_btc_data_path,
             test_data_path=args.test_btc_data_path,
             window_size=args.window_size,
             forcast_window=args.forcast_window)
    elif args.model == "advanced":
        advanced(poly_degree=args.poly_degree,
             train_btc_data_path=args.train_btc_data_path,
             test_btc_data_path=args.test_btc_data_path,
             train_tweet_data_path=args.train_tweet_data_path,
             test_tweet_data_path=args.test_tweet_data_path,
             window_size=args.window_size,
             forcast_window=args.forcast_window)
    else:
        print(f"Unknown model: {args.model}")