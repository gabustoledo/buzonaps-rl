import numpy as np
import random
from collections import deque
from clases.DQN import DQNModel
import os
import pickle
from tensorflow.keras.models import load_model

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNModel(state_size, action_size, hidden_size, learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape(1, -1)
            next_state = np.array(next_state).reshape(1, -1)

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.update(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decompose_action(self, action):
        patient_id = (action // 9) + 1
        number = (action % 9) + 1
        return patient_id, number
    
    def load(self, nombre_archivo_modelo, nombre_archivo_estado, memory_size):
        # Cargar el modelo
        if os.path.exists(nombre_archivo_modelo):
            self.model.model = load_model(nombre_archivo_modelo)
            print("Modelo cargado con éxito.")
        else:
            print("Archivo del modelo no encontrado.")

        # Cargar el estado del agente
        if os.path.exists(nombre_archivo_estado):
            with open(nombre_archivo_estado, 'rb') as file:
                estado_agente = pickle.load(file)
            self.epsilon = estado_agente['epsilon']
            self.memory = deque(estado_agente['memory'], maxlen=memory_size)
            # Cargar cualquier otro estado necesario
            print("Estado del agente cargado con éxito.")
        else:
            print("Archivo de estado del agente no encontrado.")

    def save(self, nombre_archivo_modelo, nombre_archivo_estado):
        # Guardar el modelo
        self.model.model.save(nombre_archivo_modelo)
        print("Modelo guardado con éxito.")

        # Guardar el estado del agente
        estado_agente = {
            "epsilon": self.epsilon,
            "memory": list(self.memory)  # Convertir deque a lista para serializar
            # Añadir aquí cualquier otro estado que necesites guardar
        }

        with open(nombre_archivo_estado, 'wb') as file:
            pickle.dump(estado_agente, file)
        print("Estado del agente guardado con éxito.")