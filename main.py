import numpy as np
import argparse
import json
from clases.DQNAgent import DQNAgent
from clases.API_CONNECTION import API_CONNECTION
from clases.STATE import STATE

NUM_PACIENTES = 0
NUM_MANAGER = 0

def main(modo_recompensa, config):

    # Objeto para realizar las consultas a la API
    api_connection = API_CONNECTION()

    # Se obtiene la configuracion utilizada por el simulador
    config_sim = api_connection.get_config()

    # Se setean los valores globales
    NUM_PACIENTES = config_sim['patients_amount']
    NUM_MANAGER = config_sim['managers_amount'] * config_sim['cesfam_amount']
    DAY_MAX = int(config_sim['end_sim']/24)

    # Matriz para llevar registro de los riesgos por paciente
    matrix_risk = np.zeros((NUM_PACIENTES,DAY_MAX))
    
    # Objeto para poder calcular los nuevos estados de la matriz de estado
    state_matrix = STATE(NUM_MANAGER, NUM_PACIENTES, DAY_MAX)

    # Se inicializa la matriz de estado
    # matriz_estado = np.zeros((NUM_MANAGER, 2 + 3*NUM_PACIENTES))
    # matriz_estado = np.zeros((NUM_MANAGER, 1 + NUM_PACIENTES))
    matriz_estado = np.zeros((NUM_PACIENTES, 4))
    matriz_estado, managerPatient = state_matrix.init_state(matriz_estado)

    # Son definidas los tipo de procesos disponibles
    tipo_hora = {
        "PRE_CLASSIFY_CLINICAL_RISK" : 1, # Primero evento, evalua el riesgo clinico
        "PRE_CLASSIFY_SOCIAL_RISK" : 2, # Segundo evento, evalua el riesgo social
        "MANAGE_PATIENT" : 3, # Tercer evento, este engloba a los 4 manage siguientes (en sim no hace nada, solo elije cual de las 4 seleccionar)
        "MANAGE_MEDICAL_HOUR" : 4, # 3.1
        "MANAGE_TEST_HOUR" : 5, # 3.2
        "MANAGE_SOCIAL_HOUR" : 6, # 3.3
        "MANAGE_PSYCHO_HOUR" : 7, # 3.4
        "RE_EVALUATE_LOW_RISK" : 8, # Cuarto evento, no hace nada, solo es para ocupar el tiempo
        "RE_EVALUATE_MANAGED" : 9 # Quinto evento, no hace nada, solo es para ocupar el tiempo
    }

    tratamiento = ["PRE_CLASSIFY_CLINICAL_RISK", "PRE_CLASSIFY_SOCIAL_RISK", "MANAGE_MEDICAL_HOUR", "MANAGE_TEST_HOUR", "MANAGE_SOCIAL_HOUR", "MANAGE_PSYCHO_HOUR"]

    # A partir de la configuracion obtenida son seteados los tiempos de cada proceso
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
    state_size = len(state_matrix.flatten_state(matriz_estado))
    # action_size = NUM_PACIENTES * 9 # Es por 9 ya que son 9 las posibles acciones a realizar.
    action_size = NUM_PACIENTES # La accion me indica a que paciente debo continuar su atencion
    # dqn_agent = DQNAgent(state_size, action_size, hidden_size=64, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=5)
    dqn_agent = DQNAgent(state_size, action_size, hidden_size=16, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.996, memory_size=100, batch_size=10)

    # Se comprueba si hay un modelo almacenado
    nombre_archivo_modelo = 'modelo/config_' + str(config) + '-modo_' + str(modo_recompensa) + '-modelo_dqn.h5'
    nombre_archivo_estado = 'modelo/config_' + str(config) + '-modo_' + str(modo_recompensa) + '-estado_agente.pkl'

    # Se carga el modelo y su estado si corresponde
    # dqn_agent.load(nombre_archivo_modelo, nombre_archivo_estado, 100)

    # Time para conocer la hora del simulador
    current_clock = 0

    # Indica si se debe continuar en el bucle de ejecucion
    flag = True

    # Indica si el simulador ha enviado un nuevo estado
    new_state = True

    # Historial de tareas asignadas en formato json
    history_tasks = []

    # Historial de tareas asignada en formato numerico
    acciones_acumuladas = []

    # Matriz para historial de acciones realizadas a pacientes, y cuando fue el ultimo dia que fue realizada
    history_patients = np.zeros((NUM_PACIENTES, 9))

    # Historial de recompensa en formato json
    rewards = []

    # Lista con paciente que no autorizan, todos comienzan aqui
    no_autoriza = list(range(1, NUM_PACIENTES + 1))

    # Lista de quienes autorizan
    autoriza = []

    # Listado de procesos disponibles
    process_list = ['PRE_CLASSIFY_CLINICAL_RISK','PRE_CLASSIFY_SOCIAL_RISK','MANAGE_PATIENT','MANAGE_MEDICAL_HOUR','MANAGE_TEST_HOUR','MANAGE_SOCIAL_HOUR','MANAGE_PSYCHO_HOUR','RE_EVALUATE_LOW_RISK','RE_EVALUATE_MANAGED']
    tratamiento = process_list
    # Se tiene una recompensa global. Se le ira sumando el riesgo del paciente elegido, de elegit un paciente incativo o uno que ya elijio de le descuenta 20
    reward = 0

    # Horario libre de los manager
    matrix_horario = np.zeros((NUM_MANAGER, 1))

    # Ciclo que en cada iteracion representa un dia indicado por el simulador.
    while flag:
        # Se obtiene el estado del simulador
        json_state, responseState_json = api_connection.get_data_sim()

        # Se obtienen los pacientes que si autorizan
        autoriza_aux = api_connection.get_no_autoriza()
        autoriza_aux = [int(numero) for numero in autoriza_aux]

        # Se almacena los pacientes que si autorizan
        union_autoriza = set(autoriza) | set(autoriza_aux)
        autoriza = list(union_autoriza)

        # Estos pacientes son eliminados de la lista que no autorizan
        no_autoriza = [elemento for elemento in no_autoriza if elemento not in autoriza_aux]

        # Se consulta si el simulador ha tenido cambios o no
        if len(json_state) == 2: # Estado vacio
            try:
                if json_state[0]['vacio'] == "vaciosimulador":
                    new_state = False
            except:
                new_state = True
        elif len(json_state) == 3: # Estado final
            try:
                if json_state[0]['clock'] == "final":
                    new_state = False
                    flag = False
            except:
                new_state = True
        else: # Nuevo estado disponible
            new_state = True

        # El time interno se actualiza segun lo enviado por el simulador
        current_clock = int(json_state[-1]['clock'].split(".")[0])
        # print(json_state)

        # En caso de haber un nuevo estado
        if new_state:

            # La matriz de estado se actualiza con lo enviado por el simulador
            nueva_matriz_estado, history_tasks_aux, history_patients, matrix_risk, matrix_horario = state_matrix.update_sate(matriz_estado, json_state[:-1], tipo_hora, tipo_hora_duracion, history_patients, matrix_risk, matrix_horario)

            # Se agregan las tareas que indica el simulador como finalizadas
            history_tasks.extend(history_tasks_aux)

            # Se "aplanan" la matriz nueva y antigua, se obtiene recompensa
            flatten_state = state_matrix.flatten_state(matriz_estado)
            flatten_state_nuevo = state_matrix.flatten_state(nueva_matriz_estado)
            # reward, total_risk = state_matrix.get_recompensa(nueva_matriz_estado, modo=modo_recompensa)

            # La recompensa se almacena
            # json_reward = {
            #     "day": int(current_clock/24),
            #     "reward": reward * -1,
            #     # "total_risk": total_risk
            #     "total_risk": 0
            # }
            # if modo_recompensa == 3:
            #     json_reward["reward"] = reward
            # rewards.append(json_reward)

            # Se almacena el estado en el modelo
            # dqn_agent.remember(flatten_state, acciones_acumuladas, reward, flatten_state_nuevo, False)

            # Cambia la matriz de estado
            matriz_estado = nueva_matriz_estado

        # Tareas diarias que debe realizar el simulador
        tasks = []

        # Para cada manager se revisan las tareas a realizar
        for i in range(0, NUM_MANAGER):

            # Se obtiene el proximo horario libre del manager
            # horario_libre = matriz_estado[i, 0]
            horario_libre = matrix_horario[i, 0]

            # Se seleccionan los pacientes de ese manager
            mis_pacientes = [elemento for elemento in managerPatient if elemento["manager"] == i+1]
            mis_pacientes = [item['patient'] for item in mis_pacientes]

            # Se obtiene si hay pacientes activos para los cuales darles horas
            mis_pacientes_activos = list(set(mis_pacientes) & set(autoriza))
            
            # Si el horario libre del manager es "del dia anterior", se actualiza al inicio de la jornada
            if horario_libre < current_clock + 8:
                horario_libre = current_clock + 8
            
            # Mientras al manager le quede tiempo libre en el dia
            while (horario_libre < current_clock + 20) and (len(mis_pacientes_activos) > 0):

                # Cantidad de pacientes activos para ese manager
                cantidad_pacientes = len(mis_pacientes_activos)

                # Se obtiene el estado "aplanado"
                state = state_matrix.flatten_state(matriz_estado)

                # Se selecciona la nueva tarea a realizar, codificada.
                composite_action = dqn_agent.act(state)

                # Se decodifica la tareas para obtener el paciente y el proceso.
                # paciente_actual, process_number = dqn_agent.decompose_action(composite_action)
                paciente_actual = composite_action - 1

                # Se obtiene el paciente al que pertenece
                pos = paciente_actual%cantidad_pacientes
                paciente_actual = mis_pacientes_activos[pos]

                process_number = int((matriz_estado[int(paciente_actual) - 1, 2] + 1) % len(tratamiento))
                
                # Si el paciente pertenece al manager
                if paciente_actual in mis_pacientes_activos:

                    # Si el paciente ha autorizado
                    if not (paciente_actual in no_autoriza):

                         # Se obtiene el nombre del proceso, ya que se obtiene un numero
                        # process_actual = process_list[process_number - 1]
                        process_actual = tratamiento[process_number]

                        ultima_atencion = history_patients[int(paciente_actual) - 1, tipo_hora[process_actual] -1]

                        # Hay que valida que esa tarea no haya sido ya realizada hoy ni 2 dias anteriores
                        if (ultima_atencion < int(current_clock/24) - 2) or (ultima_atencion == 0) :

                            # Historial de acciones codificadas para entrenar modelo
                            acciones_acumuladas.append(composite_action)
                            
                            # Tiempo que tarda el proceso
                            process_time = tipo_hora_duracion[process_actual]

                            # Json de la tareas para enviar a simulador
                            task = {
                                "id_manager": i+1,
                                "id_patient": 0,
                                "process": process_actual,
                                "clock_init": horario_libre,
                                "execute_time": 0
                            }
                            task["id_patient"] = int(paciente_actual)
                            task["execute_time"] = int(process_time)

                            # Actualizacion de horario libre
                            horario_libre += process_time
                            
                            # Tareas para el siguiente dia
                            tasks.append(task)

                            # Se actualiza lista de tareas realizadas por paciente
                            history_patients[int(paciente_actual) - 1, tipo_hora[process_actual] -1] = int(current_clock/24)

                            # reward += matriz_estado[int(paciente_actual) - 1, 1]
                            reward_temp = state_matrix.get_recompensa(matriz_estado, modo_recompensa, int(paciente_actual) - 1)
                            reward += reward_temp
                            nuevo_matriz_estado = matriz_estado
                            nuevo_matriz_estado[int(paciente_actual) - 1, 2] = process_number
                            nuevo_matriz_estado[int(paciente_actual) - 1, 3] += 1

                            flatten_state = state_matrix.flatten_state(matriz_estado)
                            flatten_state_nuevo = state_matrix.flatten_state(nuevo_matriz_estado)

                            dqn_agent.remember(flatten_state, acciones_acumuladas, reward, flatten_state_nuevo, False)

                            matriz_estado = nuevo_matriz_estado
                        else: # En caso de haber elegido a alguien que ya habia elejido recientemente
                            reward -= 10

                            flatten_state = state_matrix.flatten_state(matriz_estado)
                            dqn_agent.remember(flatten_state, acciones_acumuladas, reward, flatten_state, False)

                    else: # En caso de haber elegido a alguien que no autoriza
                        reward -= 30

                        flatten_state = state_matrix.flatten_state(matriz_estado)
                        dqn_agent.remember(flatten_state, acciones_acumuladas, reward, flatten_state, False)

        # Post de las tareas
        if tasks == []:
            task = {
                "id_manager": -1,
                "id_patient": -1,
                "process": "no",
                "clock_init": -1,
                "execute_time": -1
            }
            tasks.append(task)
        print(tasks)
        responseTask = api_connection.post_task(tasks)
        
        # Entrenamiento del modelo
        dqn_agent.replay()

        current_clock += 24

    # Estado final
    json_state, responseState_json = api_connection.get_data_sim(flag = 1)

    print(json_state)

    if modo_recompensa == 1:
        name = "Riesgo del 100%"
    elif modo_recompensa == 2:
        name = "Riesgo del 40% mayor"
    elif modo_recompensa == 3:
        name = "Riesgo pacientes en medio o bajo"
    elif modo_recompensa == 4:
        name = "Riesgo promedio"

    filename = './out/rewards.json'

    # Se agrupa los history task por id_patient
    patient_processes = {}

    # Agrupar los procesos por id_patient
    for item in history_tasks:
        id_patient = item["id_patient"]
        process = item["process"]
        
        if id_patient not in patient_processes:
            patient_processes[id_patient] = []

        patient_processes[id_patient].append(process)

    # Convertir el diccionario en la lista deseada de objetos JSON
    history_tasks = [{"id_patient": id_patient, "process": processes} for id_patient, processes in patient_processes.items()]

    rewards = []
    # Se arma el arreglo de json con el rewards
    for i in range(0, DAY_MAX):
        json_aux = {
            "day": i + 1,
            "reward": 0,
            "total_risk": matrix_risk[:, i].sum()
        }
        rewards.append(json_aux)


    rewards_post = {
        "name": name,
        "mode": modo_recompensa,
        "config": config,
        "rewards": rewards,
        "history_task": history_tasks
    }

    # Se lee el archivo para agregar la nueva data
    with open(filename, 'r') as archivo:
        datos = json.load(archivo)

    # Se agrega la nueva data
    datos.append(rewards_post)

    # Se almacena el archivo con la data agregada
    with open(filename, 'w') as file:
        json.dump(datos, file, indent=4)

    # Se almacena el modelo con su estado
    # dqn_agent.save(nombre_archivo_modelo, nombre_archivo_estado)

    # rewards_response = api_connection.post_rewards(rewards_post)

    desocupado = api_connection.get_desocupado()

    np.save('matrix_risk.npy', matrix_risk)


if __name__ == '__main__':

    # Se crea analizador de argumentos
    parser = argparse.ArgumentParser(description='Script para procesar estados del simulador y entregarle tareas diarias.')

    # Argumento para modo de ejecucion
    parser.add_argument('modo', type=int, choices=[1, 2, 3, 4], help='Modo de ejecución del script. Puede ser 1, 2, 3, 4.')

    # Argumento para modo de ejecucion
    parser.add_argument('config', type=int, choices=[0, 1, 2, 3, 4], help='Modo de configuracion del script. Puede ser 1, 2, 3 o 4.')

    # Se obtienen los parametros
    args = parser.parse_args()

    # Es seleccionado el parametro para el modo
    modo = args.modo
    config = args.config

    main(modo, config)