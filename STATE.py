import numpy as np
from API_CONNECTION import API_CONNECTION

class STATE:
    def __init__(self, num_managers, num_patients):
        self.api_connection = API_CONNECTION()
        self.num_managers = num_managers
        self.num_patients = num_patients

    def flatten_state(self, matrix):
        return matrix.flatten()

    def init_state(self, matriz_estado):

        matriz_estado[:, 0] = np.arange(1, self.num_managers + 1)

        matriz_estado[:, 1] = 32

        managerPatient = self.api_connection.get_manager_patient_sim()

        for mp in managerPatient:
            manager = mp['manager']
            patient = mp['patient']

            matriz_estado[manager-1, patient+1] = 1

        return matriz_estado, managerPatient

    def update_sate(self, matriz_estado, new_state, tipo_hora):

        for ns in new_state:
            if ns['agent_type'] == 'PATIENT':
                riesgo = ns['clinical_risk']
                manager_id = ns['manager_id']
                patient_id = ns['patient_id']

                matriz_estado[manager_id-1, patient_id + self.num_patients + 1] = riesgo

            if ns['agent_type'] == 'MANAGER':
                process = ns['process']
                manager_id = ns['manager_id']
                patient_id = ns['patient_id']
                sim_clock = ns['sim_clock']

                process_id = tipo_hora[process]

                matriz_estado[manager_id-1, patient_id + 2*self.num_patients + 1] = process_id

                matriz_estado[manager_id-1, 1] = sim_clock

        return matriz_estado

    def get_recompensa(self, matriz, modo=1, porcentaje=0.4):
        if modo == 1: # Riesgo de todos
            riesgo_total = matriz[:,self.num_patients+2:-self.num_patients].sum()
            recompensa = riesgo_total * -1
            return recompensa, riesgo_total
        elif modo == 2: # Riesgo de cierto porcentaje mayor
            recompensa = matriz[:,self.num_patients+2:-self.num_patients].flatten().tolist()
            recompensa.sort(reverse=True)
            recompensa = recompensa[:self.num_patients]
            riesgo_total = sum(recompensa)

            cantidad = int(self.num_patients * porcentaje)
            recompensa = recompensa[:cantidad]

            recompensa_total = sum(recompensa)
            recompensa_total = recompensa_total * -1
            return recompensa_total, riesgo_total
