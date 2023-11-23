import matplotlib.pyplot as plt
import json

# Función para procesar los datos
def process_data(filename, max_day):
    with open(filename, 'r') as file:
        data = json.load(file)

    days, rewards, total_risks = [], [], []
    last_day = -1
    for entry in data:
        day = entry["day"]
        reward = entry["reward"]
        total_risk = entry["total_risk"]
        
        while last_day < day - 1:
            last_day += 1
            days.append(last_day)
            rewards.append(reward if last_day != day else entry["reward"])
            total_risks.append(total_risk if last_day != day else entry["total_risk"])

    # Extender los datos hasta el día máximo
    while last_day < max_day:
        last_day += 1
        days.append(last_day)
        rewards.append(rewards[-1])
        total_risks.append(total_risks[-1])

    # return days, rewards, total_risks
    return days, total_risks

# Leer los archivos y encontrar el día máximo
def find_max_day(filenames):
    max_day = -1
    for filename in filenames:
        with open(filename, 'r') as file:
            data = json.load(file)
        if data:
            max_day = max(max_day, data[-1]["day"])
    return max_day

# Lista de archivos JSON
json_files = ['rewards_total.json', 'rewards_porcentaje.json', 'rewards_medio_bajo.json', 'rewards_logaritmo.json']  # Agrega más archivos aquí

# Encontrar el día máximo
max_day = find_max_day(json_files)

# Procesar datos de todos los archivos
data_dict = {}
for file in json_files:
    days, total_risks = process_data(file, max_day)
    data_dict[file] = (days, total_risks)

# Crear un único gráfico para comparar Total Risk
plt.figure(figsize=(10, 6))
for filename, (days, total_risks) in data_dict.items():
    plt.plot(days, total_risks, label=f"Total Risk - {filename}")

plt.xlabel("Days")
plt.ylabel("Total Risk")
plt.title("Total Risk Comparison Among Different Data Sources")
plt.legend()
plt.show()

# # Encontrar el día máximo
# max_day = find_max_day(['rewards_total.json', 'rewards_porcentaje.json'])

# # Procesar datos de ambos archivos
# days_total, rewards_total, total_risks_total = process_data('rewards_total.json', max_day)
# days_porcentaje, rewards_porcentaje, total_risks_porcentaje = process_data('rewards_porcentaje.json', max_day)

# # Primer conjunto de gráficos
# plt.figure(figsize=(20, 6))

# # Gráfico de Reward y Total Risk de data_total
# plt.subplot(1, 2, 1)
# plt.plot(days_total, rewards_total, label="Reward - Total")
# plt.plot(days_total, total_risks_total, label="Total Risk - Total")
# plt.xlabel("Days")
# plt.ylabel("Values")
# plt.title("Reward and Total Risk Over Days (Total Data)")
# plt.legend()

# # Gráfico de Reward y Total Risk de data_porcentaje
# plt.subplot(1, 2, 2)
# plt.plot(days_porcentaje, rewards_porcentaje, label="Reward - Percentage")
# plt.plot(days_porcentaje, total_risks_porcentaje, label="Total Risk - Percentage")
# plt.xlabel("Days")
# plt.title("Reward and Total Risk Over Days (Percentage Data)")
# plt.legend()

# plt.show()

# # Segundo conjunto de gráficos
# plt.figure(figsize=(20, 6))

# # Gráfico para comparar Total Risk
# plt.subplot(1, 2, 1)
# plt.plot(days_total, total_risks_total, label="Total Risk - Total")
# plt.plot(days_porcentaje, total_risks_porcentaje, label="Total Risk - Percentage")
# plt.xlabel("Days")
# plt.ylabel("Total Risk")
# plt.title("Total Risk Comparison")
# plt.legend()

# # Gráfico para comparar Reward
# plt.subplot(1, 2, 2)
# plt.plot(days_total, rewards_total, label="Reward - Total")
# plt.plot(days_porcentaje, rewards_porcentaje, label="Reward - Percentage")
# plt.xlabel("Days")
# plt.ylabel("Reward")
# plt.title("Reward Comparison")
# plt.legend()

# plt.show()