import requests
import time
import json

# script consulta si hay estado disponible, de estarlo se ejecuta el RL y entrega las tareas

# The API endpoint
taskPost = "http://localhost:3000/api/rl/task"
stateGet = "http://localhost:3000/api/sim/state"

flag = True
while flag:

	# Se obtiene el estado del simulador
	while True:
		responseState = requests.get(stateGet)
		json_text = responseState.text
		json_text = json_text.replace("{\"[{","[{")
		json_text = json_text.replace("}]\":\"\"}","}]")
		json_text = json_text.replace('\\"', '"')
		responseState_json = responseState.json()
		if responseState_json != {}:
			json_state = json.loads(json_text)
			print(json_state)
			break
		else:
			time.sleep(0.5)

	# codificar estado para que sea entendible por el rl

	# Aqui debe ejecutarse el rl

	# Decodificar tareas entregadas y entregar algo entendible por el simulador

	# Post de las tareas
	responseTask = requests.post(taskPost, json=responseState_json)