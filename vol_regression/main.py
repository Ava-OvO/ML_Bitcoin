from algorithm import PolynomialVolatilityPredictor


def baseline(poly_degree=4, 
             data_path='/Users/yirongyang/Documents/GitHub/ML_Bitcoin/ML_Bitcoin/toy_btc_ohlcv.csv'
             ):
    # print("test")
     # 1. 初始化预测器，可以指定多项式的阶数
    predictor = PolynomialVolatilityPredictor(degree=poly_degree)
    
    # 2. 加载数据
    # 注意：请确保文件路径正确
    try:
        predictor.load_data(data_path)
        
        # 3. 训练模型
        predictor.train()
        
        # 4. 获取处理后的数据（包含volatility和direction）
        processed_data = predictor.get_processed_data()
        # print("处理后的数据 (前5行):")
        print("Processed data (first 5 rows):")
        print(processed_data.head())
        
        # 5. 预测未来5天的波动率
        future_volatility = predictor.predict(future_steps=5)
        # print("\n预测的未来5天波动率:")
        print("\nPredicted volatility for the next 5 days:")
        print(future_volatility)

    except FileNotFoundError:
        # print("错误: toy_btc_ohlcv.csv 文件未找到。请检查路径是否正确。")
        print("Error: toy_btc_ohlcv.csv file not found. Please check if the path is correct.")
    except Exception as e:
        # print(f"发生错误: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    model_flag = 'baseline' # 'baseline' or 'market+sentiment'
    if model_flag == 'baseline':
        baseline()
    elif model_flag == 'market+sentiment':
        #TODO
        pass
    else:
        print("invalid model, please check the model_flag variable.")