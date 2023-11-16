import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNModel:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001):
        self.model = self._build_model(state_size, action_size, hidden_size, learning_rate)

    def _build_model(self, state_size, action_size, hidden_size, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_size, input_dim=state_size, activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def update(self, state, q_values):
        self.model.fit(state, q_values, epochs=1, verbose=0)