import requests
import time
import json
import numpy as np

NUM_PACIENTES = 100
NUM_MANAGER = 10
CLOCK_MAX = 8760

# script consulta si hay estado disponible, de estarlo se ejecuta el RL y entrega las tareas

# The API endpoint
taskPost = "http://localhost:3000/api/rl/task"
stateGet = "http://localhost:3000/api/sim/state"

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

def update_state_manager(datos_simulador, matriz_estado, tiempo_total_ejecucion):
    tipo_hora = {
        "ASK_CONSENT" : 0,
        "PRE_CLASSIFY_CLINICAL_RISK" : 1,
        "PRE_CLASSIFY_SOCIAL_RISK" : 2,
        "MANAGE_PATIENT" : 3,
        "MANAGE_MEDICAL_HOUR" : 4,
        "MANAGE_TEST_HOUR" : 5,
        "MANAGE_SOCIAL_HOUR" : 6,
        "MANAGE_PSYCHO_HOUR" : 7,
        "RE_EVALUATE_LOW_RISK" : 8,
        "RE_EVALUATE_MANAGED" : 9
    }

    for evento in datos_simulador:

        if evento['agent_type'] == "MANAGER":
            paciente_id = evento['patient_id']
            manager_id = evento['manager_id']
            proceso = evento['process']
            sim_clock = evento['sim_clock']
            
            # Encuentra el índice de la fila para el paciente_id en la matriz_estado
            fila_idx = np.where(matriz_estado[:, 0] == paciente_id)[0]

            if len(fila_idx) == 0:
                # Si el paciente no está en la matriz, añadirlo
                fila_idx = np.where(matriz_estado[:, 0] == 0)[0]
                if len(fila_idx) == 0:
                    # Si no hay filas vacías, no se puede añadir un nuevo paciente
                    print(f"No hay espacio para el paciente {paciente_id}")
                    continue
                fila_idx = fila_idx[0]
                matriz_estado[fila_idx, 0] = paciente_id  # Establece el ID del paciente
            else:
                fila_idx = fila_idx[0]

            # Su tiempo de espera es 0 ya que si esta en el estado enviado es debido a que se le asignado hora en el dia actual y ya no se encuentra esperando
            matriz_estado[fila_idx, 1] = 0

            # Se actualiza el id del manager que creo la cita
            matriz_estado[fila_idx, 2] = manager_id

            # Se actualiza el clock
            matriz_estado[fila_idx, 3] = sim_clock

            # Se actualiza el ultimo proceso que se ingresa 
            matriz_estado[fila_idx, 4] = tipo_hora.get(proceso, -1)

            # se indica la cantidad de veces que ha recibido cierta hora medica
            if matriz_estado[fila_idx, 4] != -1:
                indice = int(matriz_estado[fila_idx, 4] + 5)
                
                matriz_estado[fila_idx, indice] += 1

    for i in range(0,len(matriz_estado)):
        if matriz_estado[i,0] != 0:
            diferencia = tiempo_total_ejecucion - matriz_estado[i,3]

            if diferencia < 24:
                matriz_estado[i,1] = 0
            else:
                matriz_estado[i,1] = diferencia

    return matriz_estado

def update_state_patients(datos_simulador, matriz_estado, tiempo_total_ejecucion):
    tipo_hora = {
        "ANSWER_CONSENT" : 0,
        "RECEIVE_MEDICAL_HOUR" : 1,
        "RECEIVE_TEST_HOUR" : 2,
        "RECEIVE_SOCIAL_HOUR" : 3,
        "RECEIVE_PSYCHO_HOUR" : 4,
        "ATTEND_MEDICAL_HOUR" : 5,
        "ATTEND_TEST_HOUR" : 6,
        "ATTEND_SOCIAL_HOUR" : 7,
        "ATTEND_PSYCHO_HOUR" : 8
    }

    for evento in datos_simulador:

        if evento['agent_type'] == "PATIENT":
            paciente_id = evento['patient_id']
            manager_id = evento['manager_id']
            proceso = evento['process']
            sim_clock = evento['sim_clock']

            # Encuentra el índice de la fila para el paciente_id en la matriz_estado
            fila_idx = np.where(matriz_estado[:, 0] == paciente_id)[0]

            if len(fila_idx) == 0:
                # Si el paciente no está en la matriz, añadirlo
                fila_idx = np.where(matriz_estado[:, 0] == 0)[0]
                if len(fila_idx) == 0:
                    # Si no hay filas vacías, no se puede añadir un nuevo paciente
                    print(f"No hay espacio para el paciente {paciente_id}")
                    continue
                fila_idx = fila_idx[0]
                matriz_estado[fila_idx, 0] = paciente_id  # Establece el ID del paciente
            else:
                fila_idx = fila_idx[0]

            # Su tiempo de espera es 0 ya que si esta en el estado enviado es debido a que se le asignado hora en el dia actual y ya no se encuentra esperando
            matriz_estado[fila_idx, 6] = 0

            # riesgo clinico
            matriz_estado[fila_idx, 1] = evento['clinical_risk']

            # riesgo social
            matriz_estado[fila_idx, 2] = evento['social_risk']

            # clock
            matriz_estado[fila_idx, 3] = sim_clock

            # proceso
            matriz_estado[fila_idx, 4] = tipo_hora.get(proceso, -1)

            # manager
            matriz_estado[fila_idx, 5] = manager_id

            # se indica la cantidad de veces que ha recibido cierta hora medica
            if matriz_estado[fila_idx, 4] != -1:
                indice = int(matriz_estado[fila_idx, 4] + 7)

                matriz_estado[fila_idx, indice] += 1

    for i in range(0,len(matriz_estado)):
        if matriz_estado[i,0] != 0:
            diferencia = tiempo_total_ejecucion - matriz_estado[i,3]

            if diferencia < 24:
                matriz_estado[i,6] = 0
            else:
                matriz_estado[i,6] = diferencia

    return matriz_estado



matriz_estado_pacientes = np.zeros((NUM_PACIENTES, 16))
matriz_estado_manager = np.zeros((NUM_PACIENTES, 15))
matriz_estado_pacientes[:, 0] = np.arange(1, NUM_PACIENTES + 1)
matriz_estado_manager[:, 0] = np.arange(1, NUM_PACIENTES + 1)

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
        print("no hay estado nuevo")
        try:
            if json_state[0]['vacio'] == "vaciosimulador":
                print("efectivamente vacio")
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
        matriz_estado_manager = update_state_manager(json_state[:-1], matriz_estado_manager, current_clock)
        matriz_estado_pacientes = update_state_patients(json_state[:-1], matriz_estado_pacientes, current_clock)

        # se unen las matrices para enviarla al rl
        matriz_estado_manager_sin_id = matriz_estado_manager[:, 1:]
        matriz_combinada = np.hstack((matriz_estado_pacientes, matriz_estado_manager_sin_id))

        # seccion temporal para guardar matriz
        filename = "historial_matriz_combinada.txt"

        with open(filename, 'a') as f:
            # Escribir el current_clock al inicio de la nueva sección
            f.write(f"Current Clock: {current_clock}\n")
            
            # Guardar la matriz combinada en el archivo de texto
            np.savetxt(f, matriz_combinada, fmt='%s')
            
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