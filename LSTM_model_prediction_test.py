import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

class SinusInfiniteModel:
    def __init__(self,
                 minimum=1,
                 n_features=1,
                 reps=60,
                 kama=40,
                 timeline_end=100,
                 timeline_step=0.1,
                 train_split=900,
                 lstm_units=100,
                 lstm_activation='relu',
                 dense_activation=None,
                 loss_function='mse',
                 optimizer='adam',
                 epochs=100,
                 batch_size=32):
        self.minimum = minimum
        self.n_features = n_features
        self.reps = reps
        self.kama = kama
        self.timeline_end = timeline_end
        self.timeline_step = timeline_step
        self.train_split = train_split
        self.lstm_units = lstm_units
        self.lstm_activation = lstm_activation
        self.dense_activation = dense_activation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.timeline = np.arange(0, self.timeline_end, self.timeline_step)
        self.amplitude = np.sin(self.timeline) * self.timeline

        self.bid_train = self.amplitude[:self.train_split]
        self.bid_test = self.amplitude[self.train_split:]
        self.orig_test = self.bid_test

        self.n_steps_in = int(self.minimum * self.kama / 2)
        self.n_steps_out = int(self.minimum * self.kama / 2)

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.model = self._build_model()
        self.total_points = []

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, activation=self.lstm_activation, input_shape=(self.n_steps_in, self.n_features)))
        model.add(Dense(self.n_steps_out, activation=self.dense_activation))
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        return model

    def split_sequence(self, sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequence):
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train(self):
        X_train, y_train = self.split_sequence(self.bid_train, self.n_steps_in, self.n_steps_out)

        X_train = self.x_scaler.fit_transform(X_train)
        y_train = self.y_scaler.fit_transform(y_train)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.n_features))

        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict_and_visualize(self):
        bu = None
        for cycle in range(self.reps):
            current_sequence = self.bid_test if bu is None else bu
            X_test, y_test = self.split_sequence(current_sequence, self.n_steps_in, self.n_steps_out)

            X_test = self.x_scaler.transform(X_test)
            y_test = self.y_scaler.transform(y_test)
            X_test = X_test.reshape(len(X_test), self.n_steps_in, self.n_features)

            y_pred = self.model.predict(X_test)
            y_pred = self.y_scaler.inverse_transform(y_pred)
            y_test = self.y_scaler.inverse_transform(y_test)

            pred = y_pred[:, -1]
            test = y_test[:, -1]

            plt.figure()
            plt.plot(pred, label="Predicted Values", color='orange')
            plt.title(f"Cycle {cycle + 1}: Predicted Values")
            plt.legend()
            plt.show()

            alles = np.concatenate((test, pred[-1]), axis=None)
            points = [pred[-1]]
            self.total_points.append(pred[-1])

            while len(alles) > self.kama:
                X_test, y_test = self.split_sequence(alles, self.n_steps_in, self.n_steps_out)

                X_test = self.x_scaler.transform(X_test)
                y_test = self.y_scaler.transform(y_test)
                X_test = X_test.reshape(len(X_test), self.n_steps_in, self.n_features)

                y_pred = self.model.predict(X_test)
                y_pred = self.y_scaler.inverse_transform(y_pred)
                y_test = self.y_scaler.inverse_transform(y_test)

                pred = y_pred[:, -1]
                test = y_test[:, -1]

                alles = np.concatenate((test, pred[-1]), axis=None)
                points.append(pred[-1])
                self.total_points.append(pred[-1])

            x = np.arange(len(current_sequence), len(current_sequence) + len(points), 1)
            plt.figure()
            plt.plot(current_sequence, label="Original Sequence", color='blue')
            plt.plot(x, points, label="Generated Points", color='orange')
            plt.title(f"Cycle {cycle + 1}: Sequence and Points")
            plt.legend()
            plt.show()

            bu = np.concatenate((current_sequence, points), axis=None)

        x = np.arange(len(self.bid_test), len(self.bid_test) + len(self.total_points), 1)
        plt.figure()
        plt.plot(self.bid_test, label="Original Test Sequence", color='blue')
        plt.plot(x, self.total_points, label="Total Predicted Points", color='orange')
        plt.title("Final Prediction Visualization")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = SinusInfiniteModel()
    model.train()
    model.predict_and_visualize()
