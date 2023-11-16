import requests
import time
import json
import numpy as np
import random
from DQNAgent import DQNAgent

MODO = 1 # 0 aleatorio, 1 DQN

NUM_PACIENTES = 0
NUM_MANAGER = 0
CLOCK_MAX = 0

# The API endpoint
taskPost = "http://localhost:3000/api/rl/task"
stateGet = "http://localhost:3000/api/sim/state"
managerPatientGet = "http://localhost:3000/api/sim/managerPatients"
configGet = "http://localhost:3000/api/sim/config"

def flatten_state(matrix):
    return matrix.flatten()

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
            time.sleep(0.1)
    return json_state

def get_data_sim(url, flag=0):
    while True:
        responseState = requests.get(url)
        responseState_json = responseState.json()
        response_string = json.dumps(responseState_json)
        response_string = response_string[2:-6]
        json_text = response_string
        json_text = json_text.replace('\\"', '"')
        if flag == 1 and responseState_json != {}:
            json_string = next(iter(responseState_json.keys()))
            json_state = json.loads(json_string)
            break
        elif flag == 0 and responseState_json != {}:
            json_state = json.loads(json_text)
            break
        else:
            time.sleep(0.1)
    return json_state, responseState_json

def init_state(matriz_estado, managers):

    matriz_estado[:, 0] = np.arange(1, managers + 1)

    matriz_estado[:, 1] = 32

    managerPatient = get_managerPatient_sim(managerPatientGet)

    for mp in managerPatient:
        manager = mp['manager']
        patient = mp['patient']

        matriz_estado[manager-1, patient+1] = 1

    return matriz_estado, managerPatient

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

def get_recompensa(matriz):
    recompensa = matriz[:,NUM_PACIENTES+2:-NUM_PACIENTES].sum()
    return recompensa * -1

config_sim = get_config(configGet)

NUM_PACIENTES = config_sim['patients_amount']
NUM_MANAGER = config_sim['managers_amount'] * config_sim['cesfam_amount']
CLOCK_MAX = config_sim['end_sim']

# Debo conocer una matriz que inique id paciente con el id del manager
matriz_estado = np.zeros((NUM_MANAGER, 2 + 3*NUM_PACIENTES))
matriz_estado, managerPatient = init_state(matriz_estado, NUM_MANAGER)

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

tipo_hora_duracion = {
    "ASK_CONSENT" : 0,
    "PRE_CLASSIFY_CLINICAL_RISK" : 0,
    "PRE_CLASSIFY_SOCIAL_RISK" : 0,
    "MANAGE_PATIENT" : 0,
    "MANAGE_MEDICAL_HOUR" : 0,
    "MANAGE_TEST_HOUR" : 0,
    "MANAGE_SOCIAL_HOUR" : 0,
    "MANAGE_PSYCHO_HOUR" : 0,
    "RE_EVALUATE_LOW_RISK" : 0,
    "RE_EVALUATE_MANAGED" : 0
}

tipo_hora_duracion["ASK_CONSENT"] = config_sim["answer_consent_time"]
tipo_hora_duracion["PRE_CLASSIFY_CLINICAL_RISK"] = config_sim["pre_classify_clinical_risk_time"]
tipo_hora_duracion["PRE_CLASSIFY_SOCIAL_RISK"] = config_sim["pre_classify_social_risk_time"]
tipo_hora_duracion["MANAGE_PATIENT"] = config_sim["manage_patient_time"]
tipo_hora_duracion["MANAGE_MEDICAL_HOUR"] = config_sim["manage_medical_hour_time"]
tipo_hora_duracion["MANAGE_TEST_HOUR"] = config_sim["manage_test_hour_time"]
tipo_hora_duracion["MANAGE_SOCIAL_HOUR"] = config_sim["manage_social_hour_time"]
tipo_hora_duracion["MANAGE_PSYCHO_HOUR"] = config_sim["manage_psycho_hour_time"]
tipo_hora_duracion["RE_EVALUATE_LOW_RISK"] = config_sim["re_evaluate_low_risk_time"]
tipo_hora_duracion["RE_EVALUATE_MANAGED"] = config_sim["re_evaluate_managed_time"]

# Inicializacion del modelo DQN
state_size = len(flatten_state(matriz_estado))  # Tamaño del estado aplanado
action_size = NUM_PACIENTES * 10
dqn_agent = DQNAgent(state_size, action_size)

current_clock = 0
flag = True
new_state = True
clock_sim = 0
history_tasks = []
acciones_acumuladas = []
process_list = ['ASK_CONSENT','PRE_CLASSIFY_CLINICAL_RISK','PRE_CLASSIFY_SOCIAL_RISK','MANAGE_PATIENT','MANAGE_MEDICAL_HOUR','MANAGE_TEST_HOUR','MANAGE_SOCIAL_HOUR','MANAGE_PSYCHO_HOUR','RE_EVALUATE_LOW_RISK','RE_EVALUATE_MANAGED']
second_day = False
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
    elif len(json_state) == 3:
        try:
            if json_state[0]['clock'] == "final":
                new_state = False
                flag = False
        except:
            new_state = True
    else:
        new_state = True

    # print(json_state[-1])
    clock_sim = int(json_state[-1]['clock'].split(".")[0])
    current_clock = clock_sim

    # codificar estado para que sea entendible por el rl
    if new_state:
        nueva_matriz_estado = update_sate(matriz_estado, json_state[:-1], tipo_hora)

        # Si hay nuevo estado se realiza el remember aqui, pero solo desde la segunda vuelta
        if MODO == 1 and second_day:
            dqn_agent.remember(flatten_state(matriz_estado), acciones_acumuladas, get_recompensa(nueva_matriz_estado), flatten_state(nueva_matriz_estado), False)

        matriz_estado = nueva_matriz_estado

        # seccion temporal para guardar matriz
        filename = "historial_matriz.txt"

        with open(filename, 'a') as f:
            # Escribir el current_clock al inicio de la nueva sección
            f.write(f"Current Clock: {current_clock}\n")
            
            # Guardar la matriz combinada en el archivo de texto
            np.savetxt(f, matriz_estado, fmt='%s')
            
            # Escribir una línea en blanco para separar las iteraciones
            f.write("\n")

    # Seccion para generar tareas, aqui deberia ejecutarse el dqn, ya que el deberia entregar las tareas
    # ------------------------------------------------------------
    # Aqui debe ejecutarse el rl
    tasks = []

    # Por cada manager se obtendra su prox horario libre, este horario libre debe ser si o si en el horario current_clock + 8 , current_clock + 20, de lo contrario se pasa al siguiente
    # Crearle tareas random para sus pacientes (aqui necesitare esa lista que borre), un processo aleatorio y obtener el time de cada proceso.
    for i in range(0, NUM_MANAGER):
        horario_libre = matriz_estado[i, 1]
        mis_pacientes = [elemento for elemento in managerPatient if elemento["manager"] == i+1]
        
        if horario_libre < current_clock + 8:
            horario_libre = current_clock + 8

        while horario_libre < current_clock + 20:

            # paciente y proceso son aquellos elementos que deben ser seleccionados por el rl
            if MODO == 0:
                paciente_actual = random.choice(mis_pacientes)
                process_actual = random.choice(process_list)
            if MODO == 1:
                state = flatten_state(matriz_estado)
                composite_action = dqn_agent.act(state)
                paciente_actual, process_number = dqn_agent.decompose_action(composite_action)
                process_actual = process_list[process_number - 1]
                acciones_acumuladas.append(composite_action)

            process_time = tipo_hora_duracion[process_actual]
            task = {
                "id_manager": i+1,
                "id_patient": 0,
                "process": process_actual,
                "clock_init": horario_libre,
                "execute_time": 0
            }
            if MODO == 0:
                task["id_patient"] = paciente_actual["patient"]
            if MODO == 1:
                task["id_patient"] = paciente_actual

            task["execute_time"] = process_time
            horario_libre += process_time
            history_tasks.append(task)
            tasks.append(task)
    # ------------------------------------------------------------

    # Post de las tareas
    responseTask = requests.post(taskPost, json=tasks)

    if current_clock >= CLOCK_MAX:
        flag = False

    if MODO == 1 and second_day:
        dqn_agent.replay()

    second_day = True

json_state, responseState_json = get_data_sim(stateGet,1)