"""Pseudocode description of the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

import math
import numpy
import tensorflow as tf
from typing import List

##########################
####### Helpers ##########


class AlphaZeroConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000

    self.num_sampling_moves = 30
    self.max_moves = 512  # for chess and shogi, 722 for Go.
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }


class Node(object):

  def __init__(self, depth: int, total_prob: float, game: Game, action_from_parent: int):
    self.depth = depth
    self.total_prob = total_prob
    self.game = game
    self.action_from_parent = action_from_parent
    self.children = {}
    self.visit_count = 0

  def expanded(self):
    return len(self.children) > 0


class Game(object):
  TERMINAL_REWARD = 10
  STEP_REWARD = -0.1

  def __init__(self, history=None):
    self.history = history or []
    self.child_visits = []
    self.num_actions = 4

  def terminal(self):
    # Game specific termination rules.
    pass

  def discounted_terminal_value(self, state_index):
    # TODO: currently value function is not used in playing anyway
    final_reward = TERMINAL_REWARD if self.terminal() else 0
    steps_to_end = len(self.history) - state_index
    return final_reward + steps_to_end * STEP_REWARD

  def legal_actions(self):
    # Game specific calculation of legal actions.
    return []

  def clone(self):
    return Game(list(self.history))

  def apply(self, action):
    self.history.append(action)

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.itervalues())
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.num_actions)
    ])

  def store_terminated_game_statistics(self, node, action):
    # We are sure which path is the best
    self.child_visits.append([
        1 if a == action else 0
        for a in range(self.num_actions)
    ])

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int):
    return (self.discounted_terminal_value(state_index),
            self.child_visits[state_index])


class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    move_sum = float(sum(len(g.history) for g in self.buffer))
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer])
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class Network(object):

  def inference(self, image):
    return (-1, {})  # Value, Policy

  def get_weights(self):
    # Returns the weights of this network.
    return []


class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.iterkeys())]
    else:
      return make_uniform_network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


##### End Helpers ########
##########################


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for i in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing LevinTS until it finds a solution.
def play_game(config: AlphaZeroConfig, network: Network):
  game = Game()
  while not game.terminal() and len(game.history) < config.max_moves:
    game = run_levin(config, game, network)
  return game


# Levin TS is used to find a path leading to terminal state.
# If terminal node is found, it ends this game.
# If no path is found in given budget, next action is chosen
# according to visits count.
def run_levin(config: AlphaZeroConfig, game: Game, network: Network) -> Game:
  root = Node(0, 1, game, None)
  visited = set()
  fringe = set([root])
  for _ in range(config.num_simulations):
      node = select_node_to_evaluate(fringe)
      fringe.remove(node)
      if node.game.terminal():
          for n in <path_from_root_to_terminal_node>:
              node.game.store_terminated_game_statistics(n, n.action_from_parent)
          return node.game

      visited.add(node)
      children = evaluate(node, network)
      backpropagate(node, root)
      fringe.update(children)

  # Path to terminal state was not found.
  # Use some fallback
  return fallback_1(game, root, visited, fringe, config)

def fallback_1(game, root, visited, fringe, config):
    # Use visit count.
    next_action = select_action(config, game, root)
    game.apply(next_action)
    game.store_search_statistics(root)
    return game

def fallback_2(game, root, visited, fringe, config):
    # Like `fallback_1`, but make a few steps ahead instead of just one
    pass

def fallback_3(game, root, visited, fringe, config):
    # Like `fallback_1`, but do not throw away calculated
    # `visited`, and `fringe` set (and all statistics) - remove
    # from fringe set nodes that are descendants of root's children
    # that are not selected as next action
    pass

def fallback_4(game, root, visited, fringe, config):
    # Use value function somehow. Currently value function is not used anywhere.
    # Simple example:
    _, node = max((node.value, node) for node in fringe)
    ret_game = node.game
    for n in <path_from_root_to_node>:
        ret_game.store_search_statistics(n)
    return ret_game


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.iteritems()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


def select_node_to_evaluate(fringe: Set[Node]):
  _, node = max((node.depth / node.total_prob, node) for node in fringe)
  return node


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, network: Network):
  game = node.game
  value, policy_logits = network.inference(game.make_image(-1))
  node.value = value

  # Expand the node.
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.itervalues())
  for action, p in policy.iteritems():
    child_game = game.clone()
    child_game.apply(action)

    conditional_prob = p / policy_sum
    total_prob = node.total_prob * conditional_prob
    node.children[action] = Node(node.depth + 1, total_prob, child_game, action)
  return node.children


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(node: Node, root: Node):
  while node != root:
      node = node.parent
      node.visit_count += 1

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = Network()
  optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                         config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  for image, (target_value, target_policy) in batch:
    value, policy_logits = network.inference(image)
    loss += (
        tf.losses.mean_squared_error(value, target_value) +
        tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=target_policy))

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return Network()
