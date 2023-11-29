import requests
import time
import json

class API_CONNECTION:
    def __init__(self):
        self.task_post = "http://localhost:3000/api/rl/task"
        self.state_get = "http://localhost:3000/api/sim/state"
        self.manager_patient_get = "http://localhost:3000/api/sim/managerPatients"
        self.config_get = "http://localhost:3000/api/sim/config"
        self.rewards_post = "http://localhost:3000/api/rl/rewards"
        self.desocupado = "http://localhost:3000/api/sim/desocupado"
        self.no_autoriza = "http://localhost:3000/api/sim/noautoriza"

    def get_config(self):
        responseConfig = requests.get(self.config_get)
        json_text = responseConfig.text
        json_config = json.loads(json_text)
        return json_config

    def get_manager_patient_sim(self):
        while True:
            responseManagerPatient = requests.get(self.manager_patient_get)
            json_text = responseManagerPatient.text
            responseState_json = responseManagerPatient.json()
            if responseState_json != {}:
                json_state = json.loads(json_text)
                break
            else:
                time.sleep(0.1)
        return json_state

    def get_data_sim(self, flag=0):
        while True:
            responseState = requests.get(self.state_get)
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

    def post_task(self, tasks):
        return requests.post(self.task_post, json=tasks)

    def post_rewards(self, rewards):
        return requests.post(self.rewards_post, json=rewards)

    def get_desocupado(self):
        return requests.get(self.desocupado)

    def get_no_autoriza(self):
        response = requests.get(self.no_autoriza)
        response = response.text.replace('[','').replace(']','').replace('"','').split(',')
        return response