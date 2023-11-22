import numpy as np
import argparse
from DQNAgent import DQNAgent
from API_CONNECTION import API_CONNECTION
from STATE import STATE

NUM_PACIENTES = 0
NUM_MANAGER = 0
CLOCK_MAX = 0

def main(modo_recompensa):

    # Objeto para realizar las consultas a la APIj
    api_connection = API_CONNECTION()

    # Se obtiene la configuracion utilizada por el simulador
    config_sim = api_connection.get_config()

    # Se setean los valores globales
    NUM_PACIENTES = config_sim['patients_amount']
    NUM_MANAGER = config_sim['managers_amount'] * config_sim['cesfam_amount']
    CLOCK_MAX = config_sim['end_sim']
    
    # Objeto para poder calcular los nuevos estados de la matriz de estado
    state_matrix = STATE(NUM_MANAGER, NUM_PACIENTES)

    # Se inicializa la matriz de estado
    matriz_estado = np.zeros((NUM_MANAGER, 2 + 3*NUM_PACIENTES))
    matriz_estado, managerPatient = state_matrix.init_state(matriz_estado)

    # Son definidas los tipo de procesos disponibles
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
    action_size = NUM_PACIENTES * 10 # Es por 10 ya que son 10 las posibles acciones a realizar.
    dqn_agent = DQNAgent(state_size, action_size, epsilon=1.0)

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

    # Historial de recompensa en formato json
    rewards = []

    # Listado de procesos disponibles
    process_list = ['ASK_CONSENT','PRE_CLASSIFY_CLINICAL_RISK','PRE_CLASSIFY_SOCIAL_RISK','MANAGE_PATIENT','MANAGE_MEDICAL_HOUR','MANAGE_TEST_HOUR','MANAGE_SOCIAL_HOUR','MANAGE_PSYCHO_HOUR','RE_EVALUATE_LOW_RISK','RE_EVALUATE_MANAGED']
    
    # Ciclo que en cada iteracion representa un dia indicado por el simulador.
    while flag:
        # Se obtiene el estado del simulador
        json_state, responseState_json = api_connection.get_data_sim()

        # El time interno se actualiza
        current_clock += 24

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

        # En caso de haber un nuevo estado
        if new_state:

            # La matriz de estado se actualiza con lo enviado por el simulador
            nueva_matriz_estado = state_matrix.update_sate(matriz_estado, json_state[:-1], tipo_hora)

            # Se "aplanan" la matriz nueva y antigua, se obtiene recompensa
            flatten_state = state_matrix.flatten_state(matriz_estado)
            flatten_state_nuevo = state_matrix.flatten_state(nueva_matriz_estado)
            reward, total_risk = state_matrix.get_recompensa(nueva_matriz_estado, modo=modo_recompensa)

            # La recompensa se almacena
            json_reward = {
                "day": int(current_clock/24),
                "reward": reward * -1,
                "total_risk": total_risk
            }
            rewards.append(json_reward)

            # Se almacena el estado en el modelo
            dqn_agent.remember(flatten_state, acciones_acumuladas, reward, flatten_state_nuevo, False)

            # Cambia la matriz de estado
            matriz_estado = nueva_matriz_estado

        # Tareas diarias que debe realizar el simulador
        tasks = []

        # Por cada manager se obtendra su prox horario libre, este horario libre debe ser si o si en el horario current_clock + 8 , current_clock + 20, de lo contrario se pasa al siguiente
        # Crearle tareas random para sus pacientes (aqui necesitare esa lista que borre), un processo aleatorio y obtener el time de cada proceso.

        # Para cada manager se revisan las tareas a realizar
        for i in range(0, NUM_MANAGER):

            # Se obtiene el proximo horario libre del manager
            horario_libre = matriz_estado[i, 1]

            # Se seleccionan los pacientes de ese manager
            mis_pacientes = [elemento for elemento in managerPatient if elemento["manager"] == i+1]
            
            # Si el horario libre del manager es "del dia anterior", se actualiza al inicio de la jornada
            if horario_libre < current_clock + 8:
                horario_libre = current_clock + 8
            
            # Mientras al manager le quede tiempo libre en el dia
            while horario_libre < current_clock + 20:

                # Se obtiene el estado "aplanado"
                state = state_matrix.flatten_state(matriz_estado)

                # Se selecciona la nueva tarea a realizar, codificada.
                composite_action = dqn_agent.act(state)

                # Se decodifica la tareas para obtener el paciente y el proceso.
                paciente_actual, process_number = dqn_agent.decompose_action(composite_action)

                # Se obtiene el nombre del proceso, ya que se obtiene un numero
                process_actual = process_list[process_number - 1]

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
                task["id_patient"] = paciente_actual
                task["execute_time"] = process_time

                # Actualizacion de horario libre
                horario_libre += process_time

                # Historial de tareas
                history_tasks.append(task)
                
                # Tareas para el siguiente dia
                tasks.append(task)

        # Post de las tareas
        responseTask = api_connection.post_task(tasks)
        
        # Entrenamiento del modelo
        dqn_agent.replay()

    # Estado final
    json_state, responseState_json = api_connection.get_data_sim(flag = 1)

    print(json_state)

    # filename = 'rewards.json'

    # # Escribir el JSON en el archivo
    # with open(filename, 'w') as file:
    #     json.dump(rewards, file, indent=4)

    if modo_recompensa == 1:
        name = "Riesgo del 100%"
    elif modo_recompensa == 2:
        name = "Riesgo del 40% mayor"

    rewards_post = {
        "name": name,
        "rewards": rewards
    }

    rewards_response = api_connection.post_rewards(rewards_post)


if __name__ == '__main__':

    # Se crea analizador de argumentos
    parser = argparse.ArgumentParser(description='Script para procesar estados del simulador y entregarle tareas diarias.')

    # Argumento para modo de ejecucion
    parser.add_argument('modo', type=int, choices=[1, 2], help='Modo de ejecución del script. Puede ser 1 o 2.')

    # Se obtienen los parametros
    args = parser.parse_args()

    # Es seleccionado el parametro para el modo
    modo = args.modo

    main(modo)