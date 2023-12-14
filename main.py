import numpy as np
import argparse
import json
import os
import tensorflow as tf
import random
from clases.DQNAgent import DQNAgent
from clases.API_CONNECTION import API_CONNECTION
from clases.STATE import STATE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Esto establece el nivel de registro a WARN
tf.get_logger().setLevel('ERROR')  # Esto ignora todo lo que no sea de nivel ERROR

NUM_PACIENTES = 0
NUM_MANAGER = 0

def main(modo_recompensa, config, time_mode):

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

    # Matriz para llevar registro de los riesgos por paciente
    matrix_rewards = np.zeros((1,DAY_MAX+1))
    
    # Objeto para poder calcular los nuevos estados de la matriz de estado
    state_matrix = STATE(NUM_MANAGER, NUM_PACIENTES, DAY_MAX)

    # Se inicializa la matriz de estado
    matriz_estado = np.zeros((NUM_PACIENTES, 5))
    matriz_estado, managerPatient = state_matrix.init_state(matriz_estado)

    # Son definidas los tipo de procesos disponibles
    tipo_hora = {
        "PRE_CLASSIFY_CLINICAL_RISK" : 1, # Primer evento, evalua el riesgo clinico
        "PRE_CLASSIFY_SOCIAL_RISK" : 2, # Segundo evento, evalua el riesgo social
        "MANAGE_PATIENT" : 3, # Tercer evento, este engloba a los 4 manage siguientes (en sim no hace nada, solo elije cual de las 4 seleccionar)
        "MANAGE_MEDICAL_HOUR" : 4, # 3.1
        "MANAGE_TEST_HOUR" : 5, # 3.2
        "MANAGE_SOCIAL_HOUR" : 6, # 3.3
        "MANAGE_PSYCHO_HOUR" : 7, # 3.4
        "RE_EVALUATE_LOW_RISK" : 8, # Cuarto evento, no hace nada, solo es para ocupar el tiempo
        "RE_EVALUATE_MANAGED" : 9 # Quinto evento, no hace nada, solo es para ocupar el tiempo
    }

    # A partir de la configuracion obtenida son seteados los tiempos de cada proceso
    tipo_hora_duracion = {
        "ASK_CONSENT" : config_sim["ask_consent_time"],
        "PRE_CLASSIFY_CLINICAL_RISK" : config_sim["pre_classify_clinical_risk_time"],
        "PRE_CLASSIFY_SOCIAL_RISK" : config_sim["pre_classify_social_risk_time"],
        "MANAGE_PATIENT" : config_sim["manage_patient_time"],
        "MANAGE_MEDICAL_HOUR" : config_sim["manage_medical_hour_time"],
        "MANAGE_TEST_HOUR" : config_sim["manage_test_hour_time"],
        "MANAGE_SOCIAL_HOUR" : config_sim["manage_social_hour_time"],
        "MANAGE_PSYCHO_HOUR" : config_sim["manage_psycho_hour_time"],
        "RE_EVALUATE_LOW_RISK" : config_sim["re_evaluate_low_risk_time"],
        "RE_EVALUATE_MANAGED" : config_sim["re_evaluate_managed_time"],

        # Estos son los tiempo minimos de ejecucion        
        "ASK_CONSENT_MIN" : config_sim["ask_consent_time_min"],
        "PRE_CLASSIFY_CLINICAL_RISK_MIN" : config_sim["pre_classify_clinical_risk_time_min"],
        "PRE_CLASSIFY_SOCIAL_RISK_MIN" : config_sim["pre_classify_social_risk_time_min"],
        "MANAGE_PATIENT_MIN" : config_sim["manage_patient_time_min"],
        "MANAGE_MEDICAL_HOUR_MIN" : config_sim["manage_medical_hour_time_min"],
        "MANAGE_TEST_HOUR_MIN" : config_sim["manage_test_hour_time_min"],
        "MANAGE_SOCIAL_HOUR_MIN" : config_sim["manage_social_hour_time_min"],
        "MANAGE_PSYCHO_HOUR_MIN" : config_sim["manage_psycho_hour_time_min"],
        "RE_EVALUATE_LOW_RISK_MIN" : config_sim["re_evaluate_low_risk_time_min"],
        "RE_EVALUATE_MANAGED_MIN" : config_sim["re_evaluate_managed_time_min"],

        # Estos son los tiempo maximos de ejecucion        
        "ASK_CONSENT_MAX" : config_sim["ask_consent_time_max"],
        "PRE_CLASSIFY_CLINICAL_RISK_MAX" : config_sim["pre_classify_clinical_risk_time_max"],
        "PRE_CLASSIFY_SOCIAL_RISK_MAX" : config_sim["pre_classify_social_risk_time_max"],
        "MANAGE_PATIENT_MAX" : config_sim["manage_patient_time_max"],
        "MANAGE_MEDICAL_HOUR_MAX" : config_sim["manage_medical_hour_time_max"],
        "MANAGE_TEST_HOUR_MAX" : config_sim["manage_test_hour_time_max"],
        "MANAGE_SOCIAL_HOUR_MAX" : config_sim["manage_social_hour_time_max"],
        "MANAGE_PSYCHO_HOUR_MAX" : config_sim["manage_psycho_hour_time_max"],
        "RE_EVALUATE_LOW_RISK_MAX" : config_sim["re_evaluate_low_risk_time_max"],
        "RE_EVALUATE_MANAGED_MAX" : config_sim["re_evaluate_managed_time_max"]
    }

    # Se comprueba si hay un modelo almacenado
    nombre_archivo_modelo = 'modelo/config_' + str(config) + '-modo_' + str(modo_recompensa) + '-modelo_dqn.h5'
    nombre_archivo_estado = 'modelo/config_' + str(config) + '-modo_' + str(modo_recompensa) + '-estado_agente.pkl'

    # Se carga el modelo y su estado si corresponde
    # dqn_agent.load(nombre_archivo_modelo, nombre_archivo_estado, 100)

    # Time para conocer la hora del simulador
    current_clock = 0

    # Indica si se debe continuar en el ciclo de ejecucion
    flag = True

    # Indica si el simulador ha enviado un nuevo estado
    new_state = True

    # Historial de tareas asignadas en formato json
    history_tasks = []

    # Historial de tareas asignada en formato numerico
    acciones_acumuladas = []

    # Matriz para historial de acciones realizadas a pacientes, y cuando fue el ultimo dia que fue realizada
    history_patients = np.zeros((NUM_PACIENTES, 1))

    # Historial de recompensa en formato json
    rewards = []

    # Lista con paciente que no autorizan, todos comienzan aqui
    no_autoriza = list(range(1, NUM_PACIENTES + 1))

    # Lista de quienes autorizan
    autoriza = []
    # Se obtienen los pacientes que participan en el proceso
    autoriza = api_connection.get_no_autoriza()
    autoriza = [int(numero) for numero in autoriza]
    autoriza.sort()

    autoriza = list(range(1, NUM_PACIENTES + 1))

    # Se eliminan de la lista de no autoriza
    no_autoriza = [elemento for elemento in no_autoriza if elemento not in autoriza]

    # Listado de procesos disponibles
    process_list = ['PRE_CLASSIFY_CLINICAL_RISK','PRE_CLASSIFY_SOCIAL_RISK','MANAGE_PATIENT','MANAGE_MEDICAL_HOUR','MANAGE_TEST_HOUR','MANAGE_SOCIAL_HOUR','MANAGE_PSYCHO_HOUR','RE_EVALUATE_LOW_RISK','RE_EVALUATE_MANAGED']
    process_list = ['PRE_CLASSIFY_CLINICAL_RISK','PRE_CLASSIFY_SOCIAL_RISK','MANAGE_PATIENT','MANAGE_MEDICAL_HOUR','MANAGE_TEST_HOUR','MANAGE_SOCIAL_HOUR','MANAGE_PSYCHO_HOUR','RE_EVALUATE_MANAGED']
    tratamiento = process_list
    process_list_low = ['PRE_CLASSIFY_CLINICAL_RISK','PRE_CLASSIFY_SOCIAL_RISK','RE_EVALUATE_LOW_RISK']
    process_list_medium_high = ['PRE_CLASSIFY_CLINICAL_RISK','PRE_CLASSIFY_SOCIAL_RISK','MANAGE_PATIENT','MANAGE_MEDICAL_HOUR','MANAGE_TEST_HOUR','MANAGE_SOCIAL_HOUR','MANAGE_PSYCHO_HOUR','RE_EVALUATE_MANAGED']
    
    # Se tiene una recompensa global. Se le ira sumando el riesgo del paciente elegido, de elegit un paciente incativo o uno que ya elijio de le descuenta 20
    reward = 0

    # Inicializacion del modelo DQN
    state_size = len(state_matrix.flatten_state(matriz_estado))
    action_size = len(autoriza) # La accion me indica a que paciente debo continuar su atencion
    dqn_agent = DQNAgent(state_size, action_size, hidden_size=16, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.996, memory_size=100, batch_size=10)

    # Horario libre de los manager
    matrix_horario = np.zeros((NUM_MANAGER, 1))

    # Ciclo que en cada iteracion representa un dia indicado por el simulador.
    while flag:
        # Se obtiene el estado del simulador
        json_state, responseState_json = api_connection.get_data_sim()

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
        dia = int(current_clock/24)

        # En caso de haber un nuevo estado
        if new_state:

            # La matriz de estado se actualiza con lo enviado por el simulador
            nueva_matriz_estado, history_tasks_aux, history_patients, matrix_risk, matrix_horario = state_matrix.update_sate(matriz_estado, json_state[:-1], tipo_hora, history_patients, matrix_risk, matrix_horario)

            # Se agregan las tareas que indica el simulador como finalizadas
            history_tasks.extend(history_tasks_aux)

            # Cambia la matriz de estado
            matriz_estado = nueva_matriz_estado

        # Tareas diarias que debe realizar el simulador
        tasks = []

        # Para cada manager se revisan las tareas a realizar
        for i in range(0, NUM_MANAGER):

            # Se obtiene el proximo horario libre del manager
            horario_libre = matrix_horario[i, 0]
            
            # Si el horario libre del manager es "del dia anterior", se actualiza al inicio de la jornada
            if horario_libre < current_clock + 8:
                horario_libre = current_clock + 8

            # Lista que me indicara aquellos pacientes que no deben ser atendidos este dia
            pacientes_no_disponibles = []
            
            # Mientras al manager le quede tiempo libre en el dia, tenga pacientes activos y tenga pacientes disponibles para este dia
            while (horario_libre < current_clock + 20) and (len(pacientes_no_disponibles) != len(autoriza)):

                # Se obtiene el estado "aplanado"
                state = state_matrix.flatten_state(matriz_estado)

                # Se selecciona la nueva tarea a realizar, codificada.
                composite_action = dqn_agent.act(state)
                
                # Id del paciente actual
                id_current_patient = autoriza[int(composite_action) - 1]

                # Riesgo del paciente actual
                risk_current_patient = matriz_estado[id_current_patient-1, 1]

                # Eleccion del tratamiento segun riesgo del paciente
                if risk_current_patient <= 10:
                    tratamiento = process_list_low
                else:
                    tratamiento = process_list_medium_high

                tratamiento = process_list

                # El id del proceso siguiente en el tratamiento que se debe realizar
                process_number = int((matriz_estado[id_current_patient - 1, 2] + 1) % len(tratamiento))

                # Se obtiene el nombre del proceso, ya que se tiene solo el id
                process_actual = tratamiento[process_number]

                # El dia en que tuvo la ultima atencion el paciente
                ultima_atencion = history_patients[id_current_patient - 1, 0]

                # Hay que validar que esa tarea no haya sido ya realizada hoy ni 1 dias anteriores
                if (ultima_atencion < int(current_clock/24) - 1) or (ultima_atencion == 0):

                    # Historial de acciones codificadas para entrenar modelo
                    acciones_acumuladas.append(composite_action)
                    if time_mode == 0:
                        # Tiempo que tarda el proceso
                        process_time = tipo_hora_duracion[process_actual]
                    elif time_mode == 1:
                        min_val = tipo_hora_duracion[process_actual + "_MIN"]  # Tiempo minimo
                        max_val = tipo_hora_duracion[process_actual + "_MAX"]  # tiempo maximo

                        # Calculo de la media
                        mu = (min_val + max_val) / 2

                        # Elegir una desviación estándar como un porcentaje del rango
                        sigma = (max_val - min_val) * 0.4

                        # Se inicializa el número aleatorio
                        random_number = -1

                        # Se genera el numero aleatorio obligando a que sea mayor a 0
                        while random_number < 0:
                            random_number = round(random.normalvariate(mu, sigma), 2)

                        process_time = random_number

                    # Json de la tareas para enviar a simulador
                    task = {
                        "id_manager": i+1,
                        "id_patient": int(id_current_patient),
                        "process": process_actual,
                        "clock_init": horario_libre,
                        "execute_time": process_time
                    }

                    # Actualizacion de horario libre
                    horario_libre += process_time
                            
                    # Se agrega tarea que sera enviada al simulador
                    tasks.append(task)

                    # Se actualiza lista de tareas realizadas por paciente
                    history_patients[id_current_patient - 1, 0] = int(current_clock/24)

                    # Se obtiene la recompensa luego de indicar que se debe atender al paciente
                    reward_temp = state_matrix.get_recompensa(matriz_estado, modo_recompensa, id_current_patient - 1, dia)
                    reward += reward_temp
                    nuevo_matriz_estado = matriz_estado
                    nuevo_matriz_estado[id_current_patient - 1, 2] = process_number
                    nuevo_matriz_estado[id_current_patient - 1, 3] += 1

                    # Se actualiza el reward historico
                    matrix_rewards[0, int(current_clock/24)] += reward_temp

                    flatten_state = state_matrix.flatten_state(matriz_estado)
                    flatten_state_nuevo = state_matrix.flatten_state(nuevo_matriz_estado)

                    dqn_agent.remember(flatten_state, acciones_acumuladas, reward, flatten_state_nuevo, False)

                    matriz_estado = nuevo_matriz_estado

                    # Se indica que el paciente ya ha sido atendido hoy
                    if not id_current_patient in pacientes_no_disponibles:
                        pacientes_no_disponibles.append(id_current_patient)
                else: # En caso de haber elegido a alguien que ya habia elejido recientemente
                    reward -= 5

                    # Se actualiza el reward historico
                    matrix_rewards[0, int(current_clock/24)] -= 5
                            
                    if not id_current_patient in pacientes_no_disponibles:
                        pacientes_no_disponibles.append(id_current_patient)

                    flatten_state = state_matrix.flatten_state(matriz_estado)
                    dqn_agent.remember(flatten_state, acciones_acumuladas, reward, flatten_state, False)

        # Post de las tareas
        # En el caso en que no hay tareas para el actual dia, se le indica al simulador una tarea nula
        if tasks == []:
            task = {
                "id_manager": -1,
                "id_patient": -1,
                "process": "no",
                "clock_init": -1,
                "execute_time": -1
            }
            tasks.append(task)
        responseTask = api_connection.post_task(tasks)
        
        # Entrenamiento del modelo
        dqn_agent.replay()

        current_clock += 24

    # Estado final
    json_state, responseState_json = api_connection.get_data_sim(flag = 1)

    if modo_recompensa == 1:
        name = "Riesgo medio o alto"
    elif modo_recompensa == 2:
        name = "Riesgo alto"
    elif modo_recompensa == 3:
        name = "Riesgo bajo, medio o alto"
    elif modo_recompensa == 4:
        name = "Riesgo bajo, medio o alto, con cierta ponderacion"
    elif modo_recompensa == 5:
        name = "Riesgo del paciente mas tiempo de espera"

    if time_mode == 0:
        filename = './out/rewards.json'
    elif time_mode == 1:
        filename = './out/rewards_time.json'

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
            "reward": matrix_rewards[0, i],
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

    file_name = f"matrix_risk_{modo_recompensa}_{time_mode}"
    full_path = os.path.join("out/matrix_risk", file_name)

    counter = 1
    new_full_path = f"{full_path}.npy"
    while os.path.exists(new_full_path):
        new_full_path = f"{full_path}_{counter}.npy"
        counter += 1

    # Guarda la matriz
    np.save(new_full_path, matrix_risk)

    # np.save('matrix_risk/matrix_risk_'+str(modo_recompensa)+'_'+str(time)+'.npy', matrix_risk)


if __name__ == '__main__':

    # Se crea analizador de argumentos
    parser = argparse.ArgumentParser(description='Script para procesar estados del simulador y entregarle tareas diarias.')

    # Argumento para modo de ejecucion
    parser.add_argument('modo', type=int, choices=[1, 2, 3, 4, 5], help='Modo de ejecución del script. Puede ser 1, 2, 3, 4.')

    # Argumento para modo de ejecucion
    parser.add_argument('config', type=int, choices=[0, 1, 2, 3, 4], help='Modo de configuracion del script. Puede ser 1, 2, 3 o 4.')

    # Argumento para tiempo variable
    parser.add_argument('time_mode', type=int, choices=[0, 1], help='Modo de tiempo variable. Puede ser 0 desactivado o 1 activo.')

    # Se obtienen los parametros
    args = parser.parse_args()

    # Es seleccionado el parametro para el modo
    modo = args.modo
    config = args.config
    time_mode = args.time_mode

    main(modo, config, time_mode)