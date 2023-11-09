import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

# Inicializar entorno
env = gym.make('CartPole-v1')
EPISODES = 10

# Hiperparámetros
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95  # Factor de descuento
epsilon = 1.0  # Exploración
epsilon_min = 0.01
epsilon_decay = 0.995
memory_size = 2000
batch_size = 64
training_starts = 1000

# Red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

# Buffer de repetición de experiencia
memory = deque(maxlen=memory_size)

# Función para elegir acciones
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    qs = model.predict(state)
    return np.argmax(qs[0])

# Función para entrenar la red
def replay(batch_size, epsilon):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma * np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    return epsilon

rewards = []
# Bucle de entrenamiento principal
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if len(memory) > training_starts:
            epsilon = replay(batch_size, epsilon)
    rewards.append(total_reward)
    print(f"Episodio: {e}, Recompensa: {total_reward}")
    
plt.plot(rewards)
plt.title('Recompensa por episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa total')
plt.show()