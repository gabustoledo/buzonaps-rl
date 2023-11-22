import matplotlib.pyplot as plt
import json

filename = 'rewards.json'

# Leer el contenido del archivo JSON
with open(filename, 'r') as file:
    data = json.load(file)

# Suponiendo que 'data' es tu lista de JSON
# data = [...]  # reemplaza esto con tus datos

# Inicializar listas
days = []
rewards = []
total_risks = []

# Rellenar los datos faltantes
last_day = -1
for entry in data:
    day = entry["day"]
    reward = entry["reward"]
    total_risk = entry["total_risk"]
    
    # Rellenar los días faltantes
    while last_day < day - 1:
        last_day += 1
        days.append(last_day)
        rewards.append(reward if last_day != day else entry["reward"])
        total_risks.append(total_risk if last_day != day else entry["total_risk"])

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(days, rewards, label="Reward")
plt.plot(days, total_risks, label="Total Risk")
plt.xlabel("Days")
plt.ylabel("Values")
plt.title("Reward and Total Risk Over Days")
plt.legend()
plt.show()
