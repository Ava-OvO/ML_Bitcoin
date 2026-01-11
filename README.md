
- `data/`: Contains the raw data and scripts for data generation and preprocessing. The `01_market_only_dataset.ipynb` notebook in this directory also contains work on classification tasks.
- `vol_regression/`: Contains the main logic for the volatility regression task, including the baseline and advanced models.

## Dependencies

The project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm
- catboost (implied by `catboost_info` directories)

You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib tqdm catboost
```

## How to Run

1.  **Generate the training and testing data:**
    
    Run the `data_generate.py` script to split the raw data into training and testing sets.
    
    ```bash
    python data/data_generate.py
    ```
    
2.  **Run the volatility regression models:**
    
    Use `vol_regression/main.py` to train and evaluate the models. You can choose between the `baseline` and `advanced` models using the `--model` argument.
    
    **Baseline Model:**
    
    ```bash
    python vol_regression/main.py --model baseline --train_btc_data_path data/btc_train.csv --test_btc_data_path data/btc_test.csv
    ```
    
    **Advanced Model:**
    
    ```bash
    python vol_regression/main.py --model advanced --train_btc_data_path data/btc_train.csv --test_btc_data_path data/btc_test.csv --train_tweet_data_path data/tweet_train.csv --test_tweet_data_path data/tweet_test.csv
    ```
    
    You can also specify other parameters such as `--poly_degree`, `--window_size`, and `--forcast_window`.
    

## Models

### Baseline Model

The baseline model (`PolynomialVolatilityPredictor`) uses a polynomial regression approach based solely on historical Bitcoin price data (High, Low, Close) to predict future volatility.

### Advanced Model

The advanced model (`PolynomialMultiModalVolatilityPredictor`) extends the baseline model by incorporating sentiment scores from tweets as an additional feature. This multi-modal approach aims to improve prediction accuracy by considering market sentiment.

The results of the predictions are saved as plots in the `vol_regression` directory.