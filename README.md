
# LSTM Time Series Prediction

This project demonstrates the use of an **LSTM (Long Short-Term Memory)** neural network to perform time series prediction. The model is trained on a synthetic dataset, and predictions are made iteratively to extend the forecast into the future.

---

## Features

- Train a model to predict sinusoidal sequences using LSTM layers.
- Customize key parameters like sequence length, number of LSTM units, and dataset division.
- Visualize predictions alongside original data for evaluation.

## Parameters

The class `SinusInfiniteModel` supports a variety of customizable parameters:

### Initialization Parameters

- **`minimum`**: Determines the sequence steps used for training and prediction.
- **`n_features`**: Specifies the number of features for each input timestep.
- **`reps`**: Defines the number of prediction cycles to perform.
- **`kama`**: Sets the threshold for processing sequence points during iterative prediction.
- **`timeline_end`**: Specifies the end point of the sinusoidal timeline.
- **`timeline_step`**: Determines the increment between timeline points.
- **`train_split`**: Specifies the split point between training and testing datasets.
- **`lstm_units`**: Defines the number of units (neurons) in the LSTM layer, controlling its capacity.
- **`lstm_activation`**: Specifies the activation function for the LSTM layer.
- **`dense_activation`**: Specifies the activation function for the Dense layer.
- **`loss_function`**: Defines the loss metric for training the model.
- **`optimizer`**: Specifies the optimization algorithm used for training.
- **`epochs`**: Defines the number of iterations over the training dataset.
- **`batch_size`**: Specifies the number of samples processed in each training batch.

## How It Works

1. **Data Generation**: A sinusoidal sequence is generated based on the specified timeline parameters.
2. **Training**: The model is trained on a portion of the data to learn the underlying pattern.
3. **Prediction**: Iterative predictions are made, extending beyond the training data.
4. **Visualization**: The predictions and original data are plotted for comparison.

## Usage

### Training the Model

```python
model = SinusInfiniteModel()
model.train()
```

### Making Predictions and Visualizing

```python
model.predict_and_visualize()
```

## Visualization

The model generates plots to compare the predicted values with the actual data:
- Cycle-by-cycle predicted values are plotted in orange.
- The original sequence is plotted in blue.
- Legends and titles are included for clarity.

## Dependencies

The project relies on the following libraries:
- `numpy`
- `matplotlib`
- `sklearn`
- `tensorflow`

Install the dependencies using:
```bash
pip install numpy matplotlib scikit-learn tensorflow
```

## Future Improvements

- Add real-world datasets for training and evaluation.
- Introduce hyperparameter tuning for better model performance.
- Save and load trained models for reuse.
- Implement a GUI for better visualization and interaction.

---

