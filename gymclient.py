import pickle
import numpy
import requests


# [TODO] Error handling for if server is down
class Environment:
    def __init__(self, address, port, ssl = False):
        self.address = address
        self.port = port
        protocol = "https://" if ssl else "http://"
        self.server = protocol + address + ":" + str(port)        

    ##
    # Helper Functions
    ##
    def get_environment_name(self):
        r = requests.get(self.server + "/environment")
        return r.text
    def get_state(self):
        r = requests.get(self.server + "/state")
        return pickle.loads(r.content)
    def get_reward(self):
        r = requests.get(self.server + "/reward")
        return float(r.text)
    def get_score(self):
        r = requests.get(self.server + "/reward", params = {'all':''})
        return float(r.text)
    def get_done(self):
        r = requests.get(self.server + "/done")
        return r.text == "True"
    def get_info(self):
        r = requests.get(self.server + "/info")
        return r.json()
    
    ##
    # Common API
    ##
    def reset(self):
        r = requests.get(self.server + "/reset")
        return pickle.loads(r.content)
    def step(self, action):
        r = requests.post(self.server + "/action", data={'id': action})
        content = r.json()
        return self.get_state(), float(content['reward']), content['done'] == "True", content['info']
    
# env = Environment("127.0.0.1", 5000)