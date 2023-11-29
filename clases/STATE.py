import numpy as np
from clases.API_CONNECTION import API_CONNECTION

class STATE:
    def __init__(self, num_managers, num_patients):
        self.api_connection = API_CONNECTION()
        self.num_managers = num_managers
        self.num_patients = num_patients

    def flatten_state(self, matrix):
        return matrix.flatten()

    def init_state(self, matriz_estado):

        # Id del manager
        # matriz_estado[:, 0] = np.arange(1, self.num_managers + 1)

        # Horario libre
        # matriz_estado[:, 1] = 32
        matriz_estado[:, 0] = 32

        # Se obtiene la relacion manager-patient
        managerPatient = self.api_connection.get_manager_patient_sim()

        # for mp in managerPatient:
        #     manager = mp['manager']
        #     patient = mp['patient']

        #     # Se indica si el patient pertenece al manager
        #     matriz_estado[manager-1, patient+1] = 1

        return matriz_estado, managerPatient

    def update_sate(self, matriz_estado, new_state, tipo_hora, tipo_hora_duracion, history_patients):
        history_task = []
        for ns in new_state:

            # Se obtiene el riesgo del paciente
            if ns['agent_type'] == 'PATIENT':
                riesgo = ns['clinical_risk']
                manager_id = ns['manager_id']
                patient_id = ns['patient_id']

                # matriz_estado[manager_id-1, patient_id + self.num_patients + 1] = riesgo
                matriz_estado[manager_id-1, patient_id] = riesgo

            # Se registra las horas atendidas
            if ns['agent_type'] == 'MANAGER':
                if not (ns['process'] == "ASK_CONSENT"):
                    process = ns['process']
                    manager_id = ns['manager_id']
                    patient_id = ns['patient_id']
                    sim_clock = ns['sim_clock']

                    process_id = tipo_hora[process]

                    # matriz_estado[manager_id-1, patient_id + 2*self.num_patients + 1] = process_id
                    
                    nueva_hora_libre = sim_clock #+ tipo_hora_duracion[process]
                    # if matriz_estado[manager_id-1, 1] < nueva_hora_libre:
                    #     matriz_estado[manager_id-1, 1] = nueva_hora_libre
                    if matriz_estado[manager_id-1, 0] < nueva_hora_libre:
                        matriz_estado[manager_id-1, 0] = nueva_hora_libre

                    json_aux = {
                            "id_patient": str(ns["patient_id"]) + "_" + str(ns["manager_id"]),
                            # "process": ns["process"] + "_" + str(sim_clock) + "_" + str(nueva_hora_libre) + "_" + str(matriz_estado[manager_id-1, 1])
                            "process": ns["process"] + "_" + str(sim_clock) + "_" + str(nueva_hora_libre) + "_" + str(matriz_estado[manager_id-1, 0])
                        }
                    
                    history_task.append(json_aux)

                    history_patients[patient_id - 1, process_id - 1] = int(nueva_hora_libre/24)

        return matriz_estado, history_task, history_patients

    def get_recompensa(self, matriz, modo=1, porcentaje=0.4):
        if modo == 1: # Riesgo de todos
            # riesgo_total = matriz[:,self.num_patients+2:-self.num_patients].sum()
            riesgo_total = matriz[:,1:].sum()
            recompensa = riesgo_total * -1
            return recompensa, riesgo_total
        elif modo == 2: # Riesgo de cierto porcentaje mayor
            # riesgo = matriz[:,self.num_patients+2:-self.num_patients].flatten().tolist()
            riesgo = matriz[:,1:].flatten().tolist()
            riesgo.sort(reverse=True)
            riesgo = riesgo[:self.num_patients]
            riesgo_total = sum(riesgo)

            cantidad = int(self.num_patients * porcentaje)
            riesgo = riesgo[:cantidad]

            recompensa_total = sum(riesgo)
            recompensa_total = recompensa_total * -1
            return recompensa_total, riesgo_total
        elif modo == 3: # Cantidad de pacientes con riesgo medio o bajo
            # riesgo = matriz[:,self.num_patients+2:-self.num_patients].flatten().tolist()
            riesgo = matriz[:,1:].flatten().tolist()
            riesgo.sort(reverse=True)
            riesgo = riesgo[:self.num_patients]
            riesgo_total = sum(riesgo)

            recompensa_umbral = self.num_patients
            for rm in riesgo:
                if rm == 30:
                    recompensa_umbral -= 1
            
            return recompensa_umbral, riesgo_total
        elif modo == 4: # Riesgo promedio
            # riesgo_total = matriz[:,self.num_patients+2:-self.num_patients].sum()
            riesgo_total = matriz[:,1:].sum()
            recompensa = (riesgo_total * -1) / self.num_patients
            return recompensa, riesgo_total

