import numpy as np
import random
from DQNAgent import DQNAgent
from API_CONNECTION import API_CONNECTION
from STATE import STATE

MODO = 1 # 0 aleatorio, 1 DQN

NUM_PACIENTES = 0
NUM_MANAGER = 0
CLOCK_MAX = 0

def main():

    api_connection = API_CONNECTION()

    config_sim = api_connection.get_config()

    NUM_PACIENTES = config_sim['patients_amount']
    NUM_MANAGER = config_sim['managers_amount'] * config_sim['cesfam_amount']
    CLOCK_MAX = config_sim['end_sim']
    
    state_matrix = STATE(NUM_MANAGER, NUM_PACIENTES)

    # Debo conocer una matriz que inique id paciente con el id del manager
    matriz_estado = np.zeros((NUM_MANAGER, 2 + 3*NUM_PACIENTES))
    matriz_estado, managerPatient = state_matrix.init_state(matriz_estado)

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
        "ASK_CONSENT" : config_sim["answer_consent_time"],
        "PRE_CLASSIFY_CLINICAL_RISK" : config_sim["pre_classify_clinical_risk_time"],
        "PRE_CLASSIFY_SOCIAL_RISK" : config_sim["pre_classify_social_risk_time"],
        "MANAGE_PATIENT" : config_sim["manage_patient_time"],
        "MANAGE_MEDICAL_HOUR" : config_sim["manage_medical_hour_time"],
        "MANAGE_TEST_HOUR" : config_sim["manage_test_hour_time"],
        "MANAGE_SOCIAL_HOUR" : config_sim["manage_social_hour_time"],
        "MANAGE_PSYCHO_HOUR" : config_sim["manage_psycho_hour_time"],
        "RE_EVALUATE_LOW_RISK" : config_sim["re_evaluate_low_risk_time"],
        "RE_EVALUATE_MANAGED" : config_sim["re_evaluate_managed_time"]
    }

    # Inicializacion del modelo DQN
    state_size = len(state_matrix.flatten_state(matriz_estado))  # Tamaño del estado aplanado
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
        json_state, responseState_json = api_connection.get_data_sim()

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
            nueva_matriz_estado = state_matrix.update_sate(matriz_estado, json_state[:-1], tipo_hora)

            # Si hay nuevo estado se realiza el remember aqui, pero solo desde la segunda vuelta
            if MODO == 1 and second_day:
                dqn_agent.remember(state_matrix.flatten_state(matriz_estado), acciones_acumuladas, state_matrix.get_recompensa(nueva_matriz_estado), state_matrix.flatten_state(nueva_matriz_estado), False)

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
                    state = state_matrix.flatten_state(matriz_estado)
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
        responseTask = api_connection.post_task(tasks)

        if MODO == 1 and second_day:
            dqn_agent.replay()

        second_day = True

    json_state, responseState_json = api_connection.get_data_sim(flag = 1)

    print(json_state)

if __name__ == '__main__':
    main()