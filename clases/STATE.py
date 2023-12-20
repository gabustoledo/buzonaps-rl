from clases.API_CONNECTION import API_CONNECTION
import numpy as np

class STATE:
    def __init__(self, num_managers, num_patients, day_max):
        self.api_connection = API_CONNECTION()
        self.num_managers = num_managers
        self.num_patients = num_patients
        self.day_max = day_max

    def flatten_state(self, matrix):
        return matrix.flatten()

    def init_state(self, matriz_estado):

        # ID del paciente
        matriz_estado[:, 0] = np.arange(1, self.num_patients + 1)

        # Riesgo inicial
        matriz_estado[:, 1] = 0

        # Parte dell proceso
        matriz_estado[:, 2] = -1

        # Cantidad de atenciones
        matriz_estado[:, 3] = 0

        # Dia de la ultima atencion
        matriz_estado[:, 4] = 0

        # Se obtiene la relacion manager-patient
        managerPatient = self.api_connection.get_manager_patient_sim()

        return matriz_estado, managerPatient

    def update_sate(self, matriz_estado, new_state, tipo_hora, history_patients, matrix_risk, matrix_horario, tasks_pendientes):
        history_task = []
        for ns in new_state:

            # Se obtiene el riesgo del paciente
            if ns['agent_type'] == 'PATIENT':
                riesgo = ns['clinical_risk']
                manager_id = ns['manager_id']
                patient_id = ns['patient_id']
                day = int(ns['sim_clock']/24)

                matriz_estado[patient_id-1, 1] = riesgo

                # Se actualiza el riesgo en la matriz de control del riesgo
                matrix_risk[patient_id-1, day] = riesgo

                # Luego se extiende el riesgo
                for d in range(day+1, self.day_max):
                    # Si ya existe un valor de riesgo diferente en el día siguiente, detener la propagación
                    if matrix_risk[patient_id-1, d] != 0:
                        break
                    matrix_risk[patient_id-1, d] = riesgo

            # Se registra las horas atendidas
            if ns['agent_type'] == 'MANAGER':
                if not (ns['process'] == "ASK_CONSENT"):
                    process = ns['process']
                    manager_id = ns['manager_id']
                    patient_id = ns['patient_id']
                    sim_clock = ns['sim_clock']
                    day = int(ns['sim_clock']/24)

                    process_id = tipo_hora[process]
                    
                    nueva_hora_libre = sim_clock #+ tipo_hora_duracion[process]
                    if matrix_horario[manager_id-1, 0] < nueva_hora_libre:
                        matrix_horario[manager_id-1, 0] = nueva_hora_libre

                    json_aux = {
                            "id_patient": str(ns["patient_id"]) + "_" + str(ns["manager_id"]),
                            "process": ns["process"] + "_" + str(sim_clock) + "_" + str(nueva_hora_libre) + "_" + str(matriz_estado[manager_id-1, 0])
                        }
                    
                    history_task.append(json_aux)

                    history_patients[patient_id - 1, 0] = int(nueva_hora_libre/24)

                    # Se actualiza el dia de su ultima atencion
                    matriz_estado[patient_id-1, 4] = day

                    # Se actualiza las tareas pendientes, eliminando los elementos que el simulador ya ha realizado
                    task_aux = {
                        "id_patient": str(patient_id),
                        "id_manager": str(manager_id),
                        "process": str(process_id-1)
                    }

                    tasks_pendientes = [objeto for objeto in tasks_pendientes if objeto != task_aux]


        return matriz_estado, history_task, history_patients, matrix_risk, matrix_horario, tasks_pendientes

    def get_recompensa(self, matriz, modo, paciente, dia):
        risk_patient = matriz[int(paciente) - 1, 1]
        if modo == 1: # Si el paciente tiene riesgo alto o medio, la recompensa aumenta
            if risk_patient >= 20:
                return risk_patient
            else:
                return 5
        elif modo == 2: # Si el paciente tiene riesgo algo, la recompensa aumenta
            if risk_patient == 30:
                return risk_patient
            else:
                return 5
        elif modo == 3: # La recompensa aumenta los mismo del riesgo del paciente
            if risk_patient == 0:
                return 5
            return risk_patient
        elif modo == 4: # Si es riesgo bajo aumenta 1, si riesgo medio 5, si riesgo alto 15
            if risk_patient == 30:
                return 20
            if risk_patient == 20:
                return 15
            if risk_patient == 10:
                return 10
            else:
                return 5
        elif modo == 5: # La recompensa es el riesgo del paciente y su tiempo de espera
            espera = dia - matriz[int(paciente) - 1, 4]
            return risk_patient + espera