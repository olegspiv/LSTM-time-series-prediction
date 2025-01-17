
# LSTM Time Series Prediction

This project demonstrates the use of an **LSTM (Long Short-Term Memory)** neural network to perform time series prediction. The model is trained on a synthetic dataset, and predictions are made iteratively to extend the forecast into the future.

---

## Features

- **Data Generation**: Simulates sinusoidal data with a varying amplitude.
- **Sequence Splitting**: Splits the time series data into input-output pairs for supervised learning.
- **LSTM Model**: Trains an LSTM model for multi-step forecasting.
- **Iterative Prediction**: Extends predictions beyond the test dataset.
- **Visualization**: Plots predictions vs actual test data and extended future predictions.

---

## Requirements

To run the project, the following Python libraries are required:
- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-learn`

You can install these dependencies using pip:
```bash
pip install numpy matplotlib tensorflow scikit-learn
```

---

## How It Works

1. **Synthetic Data Creation**:
   - The script generates a sinusoidal dataset, `y = sin(x) * x`, over a range.
   - The dataset is split into training and test sets.

2. **Sequence Splitting**:
   - Data is divided into sequences of `n_steps_in` input steps and `n_steps_out` output steps using the `split_sequence` function.

3. **Data Scaling**:
   - Min-Max scaling is applied to normalize the input and output data for efficient LSTM training.

4. **LSTM Model Training**:
   - An LSTM neural network is trained on the scaled training data for multi-step prediction.
   - The model predicts future values in the test set and beyond.

5. **Iterative Forecasting**:
   - Future points are predicted iteratively, using the last predictions as inputs for the next step.

6. **Visualization**:
   - The script produces two plots:
     - Predicted vs actual values for the test set.
     - Extended predictions beyond the test set.

---

## Usage

1. Navigate to the project directory:
   ```bash
   cd project-directory
   ```

2. Run the Python script:
   ```bash
   python LSTM_model_prediction_test.py
   ```

3. The script will train the model and display the following:
   - Predicted vs actual test data.
   - Future predictions extending beyond the test dataset.

---

## Code Highlights

- **LSTM Model**:
  ```python
  regression = Sequential()
  regression.add(LSTM(units=100, activation='relu', input_shape=(X_train.shape[1], n_features)))
  regression.add(Dense(n_steps_out))
  regression.compile(loss='mse', optimizer='adam')
  regression.fit(X_train, y_train, epochs=100, batch_size=32)
  ```

- **Iterative Prediction**:
  ```python
  while len(alles) > kama:
      X_test, y_test = split_sequence(alles, n_steps_in, n_steps_out)
      X_test = x_scaler.transform(X_test)
      X_test = X_test.reshape(len(X_test), n_steps_in, n_features)
      y_pred = regression.predict(X_test)
      alles = np.concatenate((test, pred[len(pred) - 1]), axis=None)
      points.append(pred[len(pred) - 1])
  ```

---

## Example Outputs

1. **Predicted vs Actual Test Data**:
   ![Predicted vs Actual](images/predicted_vs_actual.png)

2. **Extended Predictions**:
   ![Extended Predictions](images/extended_predictions.png)

---

## Future Improvements

- Add real-world datasets for training and evaluation.
- Introduce hyperparameter tuning for better model performance.
- Save and load trained models for reuse.
- Implement a GUI for better visualization and interaction.

---

