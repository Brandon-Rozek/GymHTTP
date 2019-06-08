import sys
import gym
from flask import Flask
from flask import request
import pickle
import json

# Make it so that it doesn't log every HTTP request
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

##
# OpenAI Gym State
##
class Environment:
    def __init__(self, environment_name):
        self.sim = gym.make(environment_name)
        self.environment_name = environment_name
        self.reset()
    def step(self, action):
        # [TODO] Check to see if 'action' is valid
        self.state, self.reward, self.done, self.info = self.sim.step(action)
        self.score += self.reward
    def reset(self):
        self.state = self.sim.reset()
        self.reward = 0
        self.score = 0
        self.done = False
        self.info = {}
    def preprocess(self, state):
        raise NotImplementedError
    def get_state(self, preprocess = False, pickleobj = False):
        state = self.state
        if preprocess:
            state = self.preprocess(state)
        if pickleobj:
            state = pickle.dumps(state)
        return state


##
# Pong Specific Environment Information
##
import cv2
class PongEnv(Environment):
    def __init__(self):
        super(PongEnv, self).__init__("PongNoFrameskip-v4")
    def preprocess(self, state):
        # Grayscale
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Crop irrelevant parts
        frame = frame[34:194, 15:145] # Crops to shape (160, 130)
        # Downsample
        frame = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)
        # Normalize
        frame = frame / 255
        return frame



env = PongEnv()

##
# Flask Environment
##
app = Flask(__name__)

@app.route('/environment', methods=['GET'])
def get_env():
    global env
    if request.args.get('shape') is not None:
        shape = {}
        shape['observation'] = env.sim.observation_space.shape
        shape['action'] = env.sim.action_space.n
        return json.dumps(shape)
    return env.environment_name

@app.route('/gym', methods=['GET'])
def get_extra_data():
    global env
    data = {}
    if request.args.get('action_space') is not None:
        data['action_space'] = env.sim.action_space
    if request.args.get('observation_space') is not None:
        data['observation_space'] = env.sim.observation_space
    if request.args.get('reward_range') is not None:
        data['reward_range'] = env.sim.reward_range
    if request.args.get('metadata') is not None:
        data['metadata'] = env.sim.metadata
    if request.args.get('action_meanings') is not None:
        data['action_meanings'] = env.sim.unwrapped.get_action_meanings()
    return pickle.dumps(data)

@app.route('/action_space', methods=['GET'])
def get_action_space():
    global env
    return pickle.dumps(env.sim.action_space)

@app.route('/observation_space', methods=['GET'])
def get_observation_space():
    global env
    return pickle.dumps(env.sim.observation_space)

@app.route('/reward_range', methods=['GET'])
def get_reward_range():
    global env
    return pickle.dumps(env.sim.reward_range)

@app.route('/metadata', methods=['GET'])
def get_metadata():
    global env
    return pickle.dumps(env.sim.metadata)

@app.route('/action_meanings', methods=['GET'])
def get_action_meanings():
    global env
    return pickle.dumps(env.sim.unwrapped.get_action_meanings())

@app.route('/state', methods=['GET'])
def get_state():
    return env.get_state(pickleobj = True, preprocess = request.args.get('preprocess') is not None)

@app.route('/reward', methods=['GET'])
def get_reward():
    global env
    if request.args.get('all') is not None:
        return str(env.score)
    else:
        return str(env.reward)

@app.route('/done', methods=['GET'])
def is_done():
    global env
    return str(env.done)

@app.route('/info', methods=['GET'])
def get_info():
    global env
    return json.dumps(env.info)

@app.route('/action', methods=['POST'])
def perform_action():
    global env
    action = int(request.form['id'])
    env.step(action)

    content = {}
    content['state'] = env.get_state(preprocess = request.args.get('preprocess') is not None)
    content['reward'] = env.reward
    content['done'] = env.done
    content['info'] = env.info
    return pickle.dumps(content)

@app.route('/reset')
def reset_env():
    global env
    env.reset()
    return env.get_state(pickleobj = True, preprocess = request.args.get('preprocess') is not None)

