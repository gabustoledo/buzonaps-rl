import json
import matplotlib.pyplot as plt

def plot_rewards(rewards_lists):
    """
    Grafica cada lista de recompensas en rewards_lists.

    :param rewards_lists: Lista de listas de recompensas, cada una conteniendo listas de diccionarios con claves 'day', 'reward', y 'total_risk'.
    """
    for i, rewards_sublist in enumerate(rewards_lists):
        plt.figure(i+1)  # Crear una nueva figura para cada conjunto de recompensas

        for rewards in rewards_sublist:
            days = [reward['day'] for reward in rewards]
            reward_values = [reward['reward'] for reward in rewards]

            plt.plot(days, reward_values, label=f"Elemento {rewards_sublist.index(rewards) + 1}")

        plt.xlabel('Día')
        plt.ylabel('Recompensa')
        plt.title(f'Gráfico de Recompensas - Modo {i+1}')
        plt.legend()

    plt.show()

def plot_total_risk(rewards_lists):
    """
    Grafica el total_risk de cada lista de recompensas en rewards_lists.

    :param rewards_lists: Lista de listas de recompensas, cada una conteniendo listas de diccionarios con claves 'day', 'reward', y 'total_risk'.
    """
    for i, rewards_sublist in enumerate(rewards_lists):
        plt.figure(i+1)  # Crear una nueva figura para cada conjunto de recompensas

        for rewards in rewards_sublist:
            days = [reward['day'] for reward in rewards]
            risk_values = [reward['total_risk'] for reward in rewards]

            plt.plot(days, risk_values, label=f"Elemento {rewards_sublist.index(rewards) + 1}")

        plt.xlabel('Día')
        plt.ylabel('Total de Riesgo')
        plt.title(f'Gráfico del Total de Riesgo - Modo {i+1}')
        plt.legend()

    plt.show()

def find_max_day(rewards_lists):
    """
    Encuentra el día máximo en una lista de listas de rewards, donde cada elemento es una lista de diccionarios.

    :param rewards_lists: Lista de listas de rewards, donde cada lista interna contiene diccionarios con la clave 'day'.
    :return: Lista con los días máximos para cada conjunto de rewards.
    """
    max_days = []
    for rewards_list in rewards_lists:
        max_day = None
        for rewards in rewards_list:
            for reward in rewards:
                # Comparar y actualizar el día máximo
                day = reward['day']
                if max_day is None or day > max_day:
                    max_day = day
        max_days.append(max_day)

    return max_days

def fill_missing_days(rewards_lists, global_max_day):
    """
    Rellena los días faltantes en las listas de recompensas hasta el día máximo global.

    :param rewards_lists: Lista de listas de recompensas, cada una conteniendo listas de diccionarios con claves 'day', 'reward', y 'total_risk'.
    :param global_max_day: El día máximo global hasta el cual rellenar los datos.
    :return: Lista de listas de recompensas con los días faltantes rellenados.
    """
    for rewards_sublist in rewards_lists:
        for rewards in rewards_sublist:
            if not rewards:
                continue

            last_day = 0
            last_reward = 0
            last_risk = 0
            filled_rewards = []

            for reward in rewards:
                while last_day < reward['day'] - 1:
                    last_day += 1
                    filled_rewards.append({
                        'day': last_day,
                        'reward': last_reward,
                        'total_risk': last_risk
                    })
                last_day = reward['day']
                last_reward = reward['reward']
                last_risk = reward['total_risk']
                filled_rewards.append(reward)

            # Rellenar hasta el día máximo global, si es necesario
            while last_day < global_max_day:
                last_day += 1
                filled_rewards.append({
                    'day': last_day,
                    'reward': last_reward,
                    'total_risk': last_risk
                })

            rewards[:] = filled_rewards

    return rewards_lists

def calculate_average_risk_per_day(rewards_list):
    """
    Calcula el promedio del total_risk por día para cada lista de recompensas en rewards_list.

    :param rewards_list: Lista de recompensas, cada una conteniendo diccionarios con claves 'day' y 'total_risk'.
    :return: Diccionario con el día como clave y el promedio de total_risk como valor.
    """
    risk_per_day = {}
    count_per_day = {}

    for rewards in rewards_list:
        for reward in rewards:
            day = reward['day']
            risk = reward['total_risk']
            if risk is None:
                risk = 0
            
            # Acumular el total_risk y contar las ocurrencias por día
            if day in risk_per_day:
                risk_per_day[day] += risk
                count_per_day[day] += 1
            else:
                risk_per_day[day] = risk
                count_per_day[day] = 1

    # Calcular el promedio de total_risk por día
    average_risk_per_day = [{'day': day, 'total_risk': round(risk_per_day[day] / count_per_day[day])} for day in risk_per_day]

    return average_risk_per_day

def plot_average_risk_per_mode(average_all_mode):
    plt.figure(figsize=(10, 6))

    # Iterar a través de cada modo y graficar sus datos
    for i, mode_data in enumerate(average_all_mode):
        days = [item['day'] for item in mode_data]
        risks = [item['total_risk'] for item in mode_data]

        plt.plot(days, risks, label=f'Modo {i + 1}')

    plt.xlabel('Día')
    plt.ylabel('Promedio de Riesgo Total')
    plt.title('Promedio de Riesgo Total por Día y Modo')
    plt.legend()
    plt.show()

# Leer el archivo JSON
with open('out/rewards.json', 'r') as file:
    data = json.load(file)

# Crear listas para almacenar los rewards de cada modo
rewards_mode_1 = []
rewards_mode_2 = []
rewards_mode_3 = []
rewards_mode_4 = []
rewards_mode_5 = []

# Iterar a través de cada objeto en el JSON
for item in data:
    mode = item['mode']
    rewards = item['rewards']
    config = item['config']

    rewards = sorted(rewards, key=lambda x: x['day'])

    if config == 3:
        # Dependiendo del modo, añadir los rewards a la lista correspondiente
        if mode == 1:
            rewards_mode_1.append(rewards)
        elif mode == 2:
            rewards_mode_2.append(rewards)
        elif mode == 3:
            rewards_mode_3.append(rewards)
        elif mode == 4:
            rewards_mode_4.append(rewards)

with open('out/rewards_simulador.json', 'r') as file:
    data_sim = json.load(file)

for item in data_sim:
    rewards = item['rewards']
    rewards = sorted(rewards, key=lambda x: x['day'])

    rewards_mode_5.append(rewards)

list_total_rewards = [rewards_mode_1, rewards_mode_2, rewards_mode_3, rewards_mode_4, rewards_mode_5]

max_days_for_each_mode = find_max_day(list_total_rewards)

global_max_day = max([day for rewards_list in list_total_rewards for rewards in rewards_list for day in [reward['day'] for reward in rewards]])

filled_rewards_lists = fill_missing_days(list_total_rewards, global_max_day)

plot_total_risk(filled_rewards_lists)

average_all_mode = []
for item in filled_rewards_lists:
    average_risk_mode = calculate_average_risk_per_day(item)
    average_all_mode.append(average_risk_mode)

plot_average_risk_per_mode(average_all_mode)