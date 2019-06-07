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
        self.observation_space, self.action_space, self.reward_range, self.metadata, self.action_meanings = self.get_initial_metadata()    

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
    def get_observation_space(self):
        r = requests.get(self.server + '/observation_space')
        return pickle.loads(r.content)
    def get_action_space(self):
        r = requests.get(self.server + '/action_space')
        return pickle.loads(r.content)
    def get_reward_range(self):
        r = requests.get(self.server + '/reward_range')
        return pickle.loads(r.content)
    def get_metadata(self):
        r = requests.get(self.server + '/metadata')
        return pickle.loads(r.content)
    def get_action_meanings(self):
        r = requests.get(self.server + '/action_meanings')
        return pickle.loads(r.content)
    def get_initial_metadata(self):
        r = requests.get(self.server + '/gym?observation_space&action_space&reward_range&metadata&action_meanings')
        content = pickle.loads(r.content)
        return content['observation_space'], content['action_space'], content['reward_range'], content['metadata'], content['action_meanings']
    
    ##
    # Common API
    ##
    def reset(self):
        r = requests.get(self.server + "/reset")
        return pickle.loads(r.content)
    def step(self, action):
        r = requests.post(self.server + "/action", data={'id': action})
        content = pickle.loads(r.content)
        return content['state'], content['reward'], content['done'], content['info']
    
# env = Environment("127.0.0.1", 5000)