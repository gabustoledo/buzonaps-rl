import requests
import time
import json
import numpy as np

NUM_PACIENTES = 0
NUM_MANAGER = 0
CLOCK_MAX = 0

# script consulta si hay estado disponible, de estarlo se ejecuta el RL y entrega las tareas

# The API endpoint
taskPost = "http://localhost:3000/api/rl/task"
stateGet = "http://localhost:3000/api/sim/state"
managerPatientGet = "http://localhost:3000/api/sim/managerPatients"
configGet = "http://localhost:3000/api/sim/config"

def get_config(url):
    responseConfig = requests.get(url)
    json_text = responseConfig.text
    json_config = json.loads(json_text)
    return json_config

def get_managerPatient_sim(url):
    while True:
        responseManagerPatient = requests.get(url)
        json_text = responseManagerPatient.text
        responseState_json = responseManagerPatient.json()
        if responseState_json != {}:
            json_state = json.loads(json_text)
            # print(json_state)
            break
        else:
            time.sleep(0.5)
    return json_state

def get_data_sim(url):
    while True:
        responseState = requests.get(url)
        json_text = responseState.text
        json_text = json_text.replace("{\"[{","[{")
        json_text = json_text.replace("}]\":\"\"}","}]")
        json_text = json_text.replace('\\"', '"')
        responseState_json = responseState.json()
        if responseState_json != {}:
            json_state = json.loads(json_text)
            # print(json_state)
            break
        else:
            time.sleep(0.5)
    return json_state, responseState_json

def init_state(matriz_estado, managers):

    matriz_estado[:, 0] = np.arange(1, managers + 1)

    matriz_estado[:, 1] = 8

    managerPatient = get_managerPatient_sim(managerPatientGet)

    for mp in managerPatient:
        manager = mp['manager']
        patient = mp['patient']

        matriz_estado[manager-1, patient+1] = 1

    return matriz_estado

def update_sate(matriz_estado, new_state, tipo_hora):

    for ns in new_state:
        if ns['agent_type'] == 'PATIENT':
            riesgo = ns['clinical_risk']
            manager_id = ns['manager_id']
            patient_id = ns['patient_id']

            matriz_estado[manager_id-1, patient_id + NUM_PACIENTES + 1] = riesgo

        if ns['agent_type'] == 'MANAGER':
            process = ns['process']
            manager_id = ns['manager_id']
            patient_id = ns['patient_id']
            sim_clock = ns['sim_clock']

            process_id = tipo_hora[process]

            matriz_estado[manager_id-1, patient_id + 2*NUM_PACIENTES + 1] = process_id

            matriz_estado[manager_id-1, 1] = sim_clock

    return matriz_estado

config_sim = get_config(configGet)

NUM_PACIENTES = config_sim['patients_amount']
NUM_MANAGER = config_sim['managers_amount'] * config_sim['cesfam_amount']
CLOCK_MAX = config_sim['end_sim']

# Debo conocer una matriz que inique id paciente con el id del manager
matriz_estado = np.zeros((NUM_MANAGER, 2 + 3*NUM_PACIENTES))
matriz_estado = init_state(matriz_estado, NUM_MANAGER)

tipo_hora = {
    "ASK_CONSENT" : 1,
    "PRE_CLASSIFY_CLINICAL_RISK" : 2,
    "PRE_CLASSIFY_SOCIAL_RISK" : 3,
    "MANAGE_PATIENT" : 4,
    "MANAGE_MEDICAL_HOUR" : 5,
    "MANAGE_TEST_HOUR" : 6,
    "MANAGE_SOCIAL_HOUR" : 7,
    "MANAGE_PSYCHO_HOUR" : 8,
    "RE_EVALUATE_LOW_RISK" : 9,
    "RE_EVALUATE_MANAGED" : 10
}

# tiempos obtenerlos desde el config lo mismo con pacientes y managers y clock

current_clock = 0
flag = True
new_state = True
clock_sim = 0
while flag:

    # Se obtiene el estado del simulador
    json_state, responseState_json = get_data_sim(stateGet)

    # Luego de obtener una respuesta del simulador, significa que ha pasado un dia, por lo que se actualiza el clock
    current_clock += 24

    # Se consulta si el simulador ha tenido cambios o no
    if len(json_state) == 2:
        try:
            if json_state[0]['vacio'] == "vaciosimulador":
                new_state = False
        except:
            new_state = True
    else:
        new_state = True
    # print(json_state[-1])
    clock_sim = int(json_state[-1]['clock'].split(".")[0])
    current_clock = clock_sim

    # codificar estado para que sea entendible por el rl
    if new_state:
        matriz_estado = update_sate(matriz_estado, json_state[:-1], tipo_hora)

        # seccion temporal para guardar matriz
        filename = "historial_matriz.txt"

        with open(filename, 'a') as f:
            # Escribir el current_clock al inicio de la nueva sección
            f.write(f"Current Clock: {current_clock}\n")
            
            # Guardar la matriz combinada en el archivo de texto
            np.savetxt(f, matriz_estado, fmt='%s')
            
            # Escribir una línea en blanco para separar las iteraciones
            f.write("\n")

    # Falta tener un historial de las tareas asignadas, hay que identificar y quedarse con aquellas que no han sido realizadas
    # consultar por aquellas e ir dejandolas al inicio de la cola a realizar.

    # Aqui debe ejecutarse el rl

    # Decodificar tareas entregadas y entregar algo entendible por el simulador

    # Post de las tareas
    responseTask = requests.post(taskPost, json=responseState_json)

    if current_clock >= CLOCK_MAX:
        flag = False