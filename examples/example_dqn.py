from gymclient import Environment
import numpy as np
import random
from collections import deque
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from copy import deepcopy
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Value(nn.Module):
  def __init__(self, state_size, action_size):
    super(Value, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    
    self.conv1 = nn.Conv2d(state_size[1], 32, kernel_size = (8, 8), stride = (4, 4))
    self.conv2 = nn.Conv2d(32, 64, kernel_size = (4, 4), stride = (2, 2))    
    self.conv3 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1))
    
    self.fc1 = nn.Linear(64 * 6 * 6, 384)
    
    self.value_fc = nn.Linear(384, 384)
    self.value = nn.Linear(384, 1)
    
    self.advantage_fc = nn.Linear(384, 384)
    self.advantage = nn.Linear(384, action_size)

  
  def forward(self, x):
    x = x.float() / 255
    # Size changes from (batch_size, 4, 80, 70) to ()
    x = F.relu(self.conv1(x))

    # Size changes from () to ()    
    x = F.relu(self.conv2(x))
    
    # Size changes from () to ()
    x = F.relu(self.conv3(x))
    
    # Size changes from (batch_size, 64, 6, 5) to (batch_size, 1920)
    x = x.view(-1, 64 * 6 * 6)
    x = F.relu(self.fc1(x))
    
    state_value = F.relu(self.value_fc(x))
    state_value = self.value(state_value)
    
    advantage = F.relu(self.advantage_fc(x))
    advantage = self.advantage(advantage)
    
    x = state_value + advantage - advantage.mean()
    
    if torch.isnan(x).any().item():
      print("WARNING NAN IN MODEL DETECTED")
    
    return x
    
class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    # The deque will only contain the last 20,000 entries
    self.memory = ReplayMemory(capacity=75000)
    self.gamma = 0.99 # Discount Rate
    self.model = Value(state_size, action_size)
    self.learning_rate = 0.0001
    self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
    ## Additional components for Fixed Q-Targets
    self.target_model = deepcopy(self.model)
    self.tau = 1e-3 # We want to adjust our network by .1% each time
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send to GPU if available
    self.model.to(self.device)
    self.target_model.to(self.device)
  
  def update_target_model(self):
    for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
  
  def remember(self, state, action, reward, next_state, done):
    self.memory.append(state, action, reward, next_state, done)
    
  def act_random(self):
    return random.randrange(self.action_size)
  
  def act(self, state):
    # Choose the best action based on what we already know
    # If all the action values for a given state is the same, then act randomly
    state = torch.from_numpy(state._force()).float().unsqueeze(0).to(self.device)
    with torch.no_grad():
      action_values = self.target_model(state).squeeze(0)
      action = self.act_random() if (action_values[0] == action_values).all() else action_values.argmax().item()
      return action


  def replay(self, batch_size):
    minibatch = self.memory.sample(batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)
    
    state_batch = torch.from_numpy(np.stack(state_batch)).float().to(self.device)
    action_batch = torch.tensor(action_batch, device = self.device)
    reward_batch = torch.tensor(reward_batch, device = self.device)
    not_done_batch = ~torch.tensor(done_batch, device = self.device)
    next_state_batch = torch.from_numpy(np.stack(next_state_batch))[not_done_batch].float().to(self.device)
    
    
    obtained_values = self.model(state_batch).gather(1, action_batch.view(batch_size, 1))

    with torch.no_grad():
      # Use the target model to produce action values for the next state
      # and the regular model to select the action
      # That way we decouple the value and action selecting processes (DOUBLE DQN)
      not_done_size = not_done_batch.sum()
      next_state_values = self.target_model(next_state_batch)
      next_best_action = self.model(next_state_batch).argmax(1)
      best_next_state_value = torch.zeros(batch_size, device = self.device)
      best_next_state_value[not_done_batch] = next_state_values.gather(1, next_best_action.view((not_done_size, 1))).squeeze(1)
      
    expected_values = (reward_batch + (self.gamma * best_next_state_value)).unsqueeze(1)

    loss = F.mse_loss(obtained_values, expected_values)

    self.optimizer.zero_grad()
    loss.backward()
    # Clip gradients
    for param in self.model.parameters():
      param.grad.data.clamp_(-1, 1)
    self.optimizer.step()
    
    self.update_target_model()


class FireResetEnv(gym.Wrapper):
  def __init__(self, env):
    """Take action on reset for environments that are fixed until firing."""
    gym.Wrapper.__init__(self, env)
    assert env.get_action_meanings()[1] == 'FIRE'
    assert len(env.get_action_meanings()) >= 3

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(1, **kwargs)
    if done:
      self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(2, **kwargs)
    if done:
      self.env.reset(**kwargs)
    return obs

  def step(self, ac, **kwargs):
    return self.env.step(ac, **kwargs)

class LazyFrames(object):
  def __init__(self, frames):
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""
    self._frames = frames
    self._out = None

  def _force(self):
    if self._out is None:
        self._out = np.stack(self._frames) # Custom change concatenate->stack
        self._frames = None
    return self._out
 
  def __array__(self, dtype=None):
    out = self._force()
    if dtype is not None:
        out = out.astype(dtype)
    return out

  def __len__(self):
    return len(self._force())

  def __getitem__(self, i):
    return self._force()[i]

class FrameStack(gym.Wrapper):
  def __init__(self, env, k):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

  def reset(self, **kwargs):
    ob = self.env.reset(**kwargs)
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action, **kwargs):
    ob, reward, done, info = self.env.step(action, **kwargs)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return LazyFrames(list(self.frames))

env = Environment("127.0.0.1", 5000)
# env = FrameStack(ProcessFrame(FireResetEnv(env)), 4)
env = FrameStack(FireResetEnv(env), 4)
# env.seed(SEED)
state_size = [1, 4, 80, 70]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32
EPISODES = 100
epsilon = 0.999

replaySkip = 4
batch_size = batch_size * replaySkip
# Now that we have some experiences in our buffer, start training
for episode_num in range(EPISODES):
  state = env.reset(preprocess = True)
  total_reward = 0
  done = False
  replaySkip = 4
  while not done:
    replaySkip = replaySkip - 1
    if np.random.rand() > epsilon:
        action = agent.act(state)
    else:
        action = agent.act_random()
    epsilon = epsilon * 0.99997
    next_state, reward, done, _ = env.step(action, preprocess = True)

    agent.remember(state, action, reward, next_state, done)
    total_reward = total_reward + reward
    state = next_state
    
    if done:
      print("episode: {}/{}, score: {}, epsilon: {}"
          .format(episode_num, EPISODES, total_reward, epsilon))
      break # We finished this episode
    
    if len(agent.memory) > batch_size and replaySkip <= 0:
      replaySkip = 4
      agent.replay(batch_size)
    
  
