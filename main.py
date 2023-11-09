from custom_env import CustomEnv
from dqn_model import DQN
from experience_buffer import ExperienceBuffer
from dqn_agent import DQNAgent
import random

def collect_experience(env, model, buffer, epsilon):
    state = env.reset()
    done = False
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()  # Acción aleatoria
        else:
            action = model.predict(state.reshape(1, -1)).argmax()
        next_state, reward, done, _ = env.step(action)
        buffer.add((state, action, reward, next_state, done))
        state = next_state

def main():
    # Configura el entorno, modelo DQN, buffer de experiencia, agente, etc.

    env = CustomEnv()
    model = DQN(env.action_space.n)
    target_model = DQN(env.action_space.n)
    buffer = ExperienceBuffer(buffer_size=10000)
    agent = DQNAgent(env, model, target_model, buffer, gamma=0.99, batch_size=32)
    
    epsilon = 0.1  # Valor inicial de epsilon
    epsilon_decay = 0.99  # Tasa de decaimiento de epsilon

    for episode in range(100):  # Número de episodios de entrenamiento
        current_state = env.reset()
        total_reward = 0

        while True:
            # epsilon = 0.1  # Puedes ajustar la exploración aquí

            # aqui debo actualizar current_state con la lectura del archivo

            # Recolecta experiencias a medida que el agente interactúa con el entorno
            print("hola")
            collect_experience(env, model, buffer, epsilon)

            action = agent.choose_action(current_state.reshape(1, -1), epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.add((current_state, action, reward, next_state, done))
            current_state = next_state
            total_reward += reward

            # Realiza el paso de entrenamiento
            agent.train_step()

            if done:
                break

        # Actualización de epsilon (exploración) o cualquier otra lógica de entrenamiento
        # Actualización de epsilon (exploración)
        epsilon *= epsilon_decay
        epsilon = max(epsilon, 0.1)

        print(f'Episodio {episode + 1}, Recompensa total: {total_reward}')

    # Cierra el entorno si es necesario
    env.close()

if __name__ == "__main__":
    main()