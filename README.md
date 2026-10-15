```markdown
# India Primary Energy Consumption Forecasting using LSTM

This repository contains a Deep Learning project that forecasts the primary energy consumption of India using a Long Short-Term Memory (LSTM) network. The model is trained on historical energy data from 1965 to 2023.

## 📊 Dataset
The dataset used for this project is the **World Energy Consumption** dataset, downloaded via `kagglehub` from Kaggle. It contains yearly energy consumption statistics across various countries and regions.

### Data Preprocessing & Preparation
1. **Extraction**: Historical data for "India" was filtered and transformed from a wide format (years as columns) into a clean, long-format time series.
2. **Time Range**: 1965 to 2023 (59 yearly records).
3. **Feature Scaling**: `MinMaxScaler` was fitted exclusively on the training subset to avoid data leakage.
4. **Supervised Learning Formulation**: The problem was framed as a supervised sequence-to-value task where the model uses a sliding window of **the previous 5 years** to predict the primary energy consumption of the next year.
5. **Data Split**: Chronological 80:20 split:
   * **Training Set**: 1965 to 2011 (42 training sequences)
   * **Test Set**: 2012 to 2023 (12 evaluation sequences)

---

## 🤖 Model Architecture

The forecasting model is built using TensorFlow/Keras and features a sequential architecture optimized for time series data:

* **Input Layer**: Sequence length of 5 with 1 feature (`Input(shape=(5, 1))`)
* **LSTM Layer**: 50 units with `tanh` activation function
* **Dense Layer**: 25 units with `relu` activation function
* **Output Layer**: 1 unit (linear projection for regression)

```python
model = Sequential([
    Input(shape=(5, 1)),
    LSTM(50, activation="tanh"),
    Dense(25, activation="relu"),
    Dense(1)
])
```

### Model Training
* **Optimizer**: Adam
* **Loss Function**: Mean Squared Error (MSE)
* **Batch Size**: 16
* **Validation**: 20% validation split on the training subset
* **Early Stopping**: Monitored validation loss with a patience of 10 epochs and restored best weights to prevent overfitting.

---

## 📈 Results and Evaluation

The model was evaluated on unseen test data ranging from **2012 to 2023**.

### Performance Metrics
* **Mean Absolute Error (MAE)**: 1.5437
* **Root Mean Squared Error (RMSE)**: 1.6563

### Actual vs. Predicted Consumption (2012–2023)

| Year | Actual Consumption (EJ/other units) | Predicted Consumption |
|------|-----------------------------------|------------------------|
| 2012 | 25.02                             | 23.47                  |
| 2013 | 25.84                             | 24.71                  |
| 2014 | 27.59                             | 25.92                  |
| 2015 | 28.55                             | 27.10                  |
| 2016 | 29.83                             | 28.33                  |
| 2017 | 30.96                             | 29.56                  |
| 2018 | 32.72                             | 30.74                  |
| 2019 | 33.54                             | 32.07                  |
| 2020 | 31.78                             | 33.23                  |
| 2021 | 34.51                             | 34.12                  |
| 2022 | 36.37                             | 34.98                  |
| 2023 | 39.02                             | 35.88                  |

### 🔮 2024 Out-of-Sample Forecast
Using the sequence of data from 2019 to 2023, the trained model forecasted India's primary energy consumption for 2024:
* **Last Recorded Year (2023)**: 39.02
* **Predicted Year (2024)**: **36.78**

---

## 🛠️ Requirements & Installation

Ensure you have the following packages installed to run the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow kagglehub
```

## 🚀 How to Run
1. Run the data ingestion block to automatically fetch the latest CSV files using `kagglehub`.
2. Execute the preprocessing cells to prepare the sliding window dataset.
3. Train the LSTM model and visualize the loss curves.
4. Evaluate the metrics and visualize predictions against actual historic values.
```
