import argparse
import datetime
import os

import gym
import numpy as np
import collections
import tensorflow as tf
from functools import partial
import time
import pandas as pd
from tensorflow.keras.layers import Dense, Softmax, Multiply, Lambda, ReLU, InputLayer, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from scipy.stats import entropy
from tensorflow.keras import backend as K

np.random.seed(1)


class GaussianActionLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.1):
        super(GaussianActionLoss, self).__init__()
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """
        :param y_true: The real action we took
        :param y_pred: Mu, Sigma of the gaussian pdf
        :param sample_weight: the advantage estimator
        :return: Energy-based loss
        """
        action, advantage = tf.unstack(y_true, axis=-1)
        action = tf.cast(action, y_pred.dtype)
        mu, sigma_sq = tf.unstack(y_pred, axis=-1)
        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))
        exp_v = log_pdf * advantage
        exp_v = K.sum(exp_v + self.alpha * entropy)
        actor_loss = -exp_v
        return actor_loss


class PN_PrevModel(tf.keras.layers.Layer):
    def __init__(self, state_size, model):
        super(PN_PrevModel, self).__init__()
        self.model = model
        self.state_size = state_size

    def build(self, n_layers):
        for layer in self.model.layers:
            layer.trainable = False
        self.model.build((None, self.state_size))
        layers_names = [self.model.layers[i].name for i in range(n_layers)]
        outs = [self.model.get_layer(layer_name).output for layer_name in layers_names]
        return tf.keras.Model(inputs=self.model.inp, outputs=outs)


class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, action_sizes, task, cont=False, name='policy_network'):
        # Estimates the policy's action distribution for a given input state
        super(PolicyNetwork, self).__init__()
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.__str__ = name
        self.task = int(task)
        self.cont = cont
        self.build_model()

    def build_model(self):
        self.inp = tf.keras.layers.Input(shape=(self.state_size), name="input_x")
        self.h1 = Dense(24, activation='relu')
        self.h2 = Dense(12, activation='relu')
        self.outputs = [Dense(max(self.action_sizes), activation='linear') for _ in range(3)]
        self.masks = tf.constant(
            [[[1 if i < n else 0 for i in range(max(self.action_sizes))]] for n in self.action_sizes], dtype=tf.float32)
        self.m = self.masks[self.task]
        if self.cont:
            self.o = tf.keras.Sequential(
                [self.outputs[task], Lambda(lambda x: Multiply()([x, self.m]))])
            self.o2 = Lambda(lambda x: tf.concat((x[:, 0], ReLU()(x[:, 1]) + 1e-5), axis=0))
        else:
            self.o = tf.keras.Sequential([self.outputs[task], Lambda(lambda x: Softmax()(x, mask=tf.squeeze(self.m)))])
            self.o2 = Lambda(lambda x: x)

    def set_task(self, task):
        self.task = int(task)
        self.m = self.masks[self.task]
        if self.cont:
            self.o = tf.keras.Sequential(
                [self.outputs[task], Lambda(lambda x: Multiply()([x, self.m]))])
            self.o2 = Lambda(lambda x: (x[:, 0], ReLU()(x[:, 1]) + 1e-5))
        else:
            self.o = tf.keras.Sequential([self.outputs[task], Lambda(lambda x: Softmax()(x, mask=tf.squeeze(self.m)))])
            self.o2 = Lambda(lambda x: x)

    def build(self, input_shape):
        super().build(input_shape)
        self.o2(self.o(self.h2(self.h1(self.inp))))

    # returns action_distribution and output for each task (at any point, only 1 task is considered)
    def call(self, x, training=None, mask=None):
        x = self.h1(x)
        x = self.h2(x)
        x = self.o(x)
        x = self.o2(x)
        return x

    #     with tf.variable_scope(name):
    #
    #         # Loss with negative log probability
    #         self.neg_log_prob_acro = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs[0],
    #                                                                             labels=self.action_discrete)
    #         self.neg_log_prob_cart = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs[1],
    #                                                                             labels=self.action_discrete)
    #         self.neg_log_prob_mount = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs[2],
    #                                                                              labels=self.action_discrete)
    #         # self.loss_acro =  tf.reduce_mean(self.neg_log_prob_acro * self.R_t)
    #         # self.loss_cart =  tf.reduce_mean(self.neg_log_prob_cart * self.R_t)
    #         # self.loss_mount = tf.reduce_mean(self.neg_log_prob * self.R_t)
    #         self.losses_pp = [tf.reduce_mean(self.neg_log_prob_acro * self.R_t),
    #                           tf.reduce_mean(self.neg_log_prob_cart * self.R_t),
    #                           tf.reduce_mean(self.neg_log_prob_mount * self.R_t)]
    #
    #         self.summary = tf.summary.merge([tf.summary.scalar('loss_acro', self.losses_pp[0]),
    #                                          tf.summary.scalar('loss_cart', self.losses_pp[1]),
    #                                          tf.summary.scalar('loss_mount', self.losses_pp[2])])
    #
    #         # self.optimizer_acro = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_acro)
    #         # self.optimizer_cart = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_cart)
    #         # self.optimizer_mount = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_mount)
    #         self.optimizers = [tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.losses_pp[0]),
    #                            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.losses_pp[1]),
    #                            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.losses_pp[2])]
    #
    # def load_weights(self, sess, graph, old_name):
    #     assign_ops = [
    #         self.W1.assign(graph.get_tensor_by_name(old_name + '/W1:0')),
    #         self.b1.assign(graph.get_tensor_by_name(old_name + '/b1:0')),
    #         self.W_acro.assign(graph.get_tensor_by_name(old_name + '/W_acro:0')),
    #         self.b_acro.assign(graph.get_tensor_by_name(old_name + '/b_acro:0')),
    #         self.W_cart.assign(graph.get_tensor_by_name(old_name + '/W_cart:0')),
    #         self.b_cart.assign(graph.get_tensor_by_name(old_name + '/b_cart:0')),
    #         self.W_mount.assign(graph.get_tensor_by_name(old_name + '/W_mount:0')),
    #         self.b_mount.assign(graph.get_tensor_by_name(old_name + '/b_mount:0'))]
    #     sess.run(assign_ops)


class ProgressiveNetwork(tf.keras.Model):
    def __init__(self, state_size, action_sizes, models, task, cont=False,
                 name='progressive_network'):
        # Estimates the policy's action distribution for a given input state
        super(ProgressiveNetwork, self).__init__()
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.models = models
        self.task = task
        self.cont = cont
        self.build_model()

    def build_model(self):
        self.model1 = self.models[0].build(2)
        self.model2 = self.models[1].build(2)
        self.h1 = Dense(12, activation='relu')
        self.a1 = Dense(12, activation='relu')
        self.h2 = Dense(6, activation='relu')
        self.a2 = Dense(6, activation='relu')
        self.output_c = Dense(max(self.action_sizes), activation='linear')
        self.masks = tf.constant(
            [[[1 if i < n else 0 for i in range(max(self.action_sizes))]] for n in self.action_sizes], dtype=tf.float32)
        self.m = self.masks[self.task]
        if self.cont:
            self.o = tf.keras.Sequential(
                [self.output_c, Lambda(lambda x: Multiply()([x, self.m]))])
            self.o2 = Lambda(lambda x: tf.concat((x[:, 0], ReLU()(x[:, 1]) + 1e-5), axis=0))
        else:
            self.o = tf.keras.Sequential([self.output_c, Lambda(lambda x: Softmax()(x, mask=tf.squeeze(self.m)))])
            self.o2 = Lambda(lambda x: x)

    def call(self, state, training=None, mask=None):
        x1 = self.h1(state)
        h1_1, h1_2 = self.model1(state)
        h2_1, h2_2 = self.model2(state)
        a1 = self.a1(tf.concat((h1_1, h2_1), axis=-1))
        x2 = self.h2(tf.concat((a1, x1), axis=-1))
        a2 = self.a2(tf.concat((h1_2, h2_2), axis=-1))
        x3 = self.o(tf.concat((a2, x2), axis=-1))
        return self.o2(x3)
        #


class CriticNetwork(tf.keras.Model):
    def __init__(self, state_size, gamma, task, name='critic_network'):
        # Estimates the value-function of the actor's policy. V^(\pi_\theta). Also used as a baseline-function.
        super(CriticNetwork, self).__init__()
        self.state_size = state_size
        self.gamma = gamma
        self.__str__ = name
        self.task = task
        self.build_model()

    def build_model(self):
        self.inp = Input(shape=(self.state_size))
        self.h1 = Dense(12, activation='relu')
        self.o = Dense(1)

    def set_task(self, task):
        self.o = Dense(1)

    def call(self, x, training=None, mask=None):
        x = self.h1(x)
        return self.o(x)

    def build(self, input_shape):
        super().build(input_shape)
        self.o(self.h1(self.inp))


class ProgressiveCriticNetwork(tf.keras.Model):
    def __init__(self, state_size, models, task, name='progressive_critic_network'):
        # Estimates the policy's action distribution for a given input state
        super(ProgressiveCriticNetwork, self).__init__()
        self.state_size = state_size
        self.models = models
        self.task = task
        self.build_model()

    def build_model(self):
        self.model1 = self.models[0].build(1)
        self.model2 = self.models[1].build(1)
        self.h1 = Dense(12, activation='relu')
        self.a1 = Dense(12, activation='relu')
        self.o = Dense(1, activation='linear')

    def call(self, state, training=None, mask=None):
        x1 = self.h1(state)
        h1_1 = self.model1(state)
        h2_1 = self.model2(state)
        a1 = self.a1(tf.concat((h1_1, h2_1), axis=-1))
        x2 = self.o(tf.concat((a1, x1), axis=-1))
        return x2
        #


# Define hyperparameters
state_size = 6
action_size = 3
mount_continuous = None
action_sizes = [3, 2, 2]

max_episodes = 5000
max_steps = 10000
discount_factor = 0.99
learning_rate_p = 0.0004
learning_rate_c = 0.0012

render = False
debug = True
# Initialize the policy network
policy = None
critic_net = None

# tensorboard variables
base_log_path = os.path.join(os.getcwd(), "logs/fit/tensorboard")
global_v_step = 0
global_p_step = 0

# Task mappings
taskID2name = {0: 'acrobot', 1: 'cartpole', 2: 'mount'}
taskName2ID = {v: k for (k, v) in taskID2name.items()}


# Start training the agent with REINFORCE algorithm

def actor_critic(episode_transitions, episode, writer=None, done=False, td_steps=1, task=0):
    """
        :param sess: The tensorflow session that is running
        :param episode_transitions: list of all the transitions in the episode
        :param episode: number of the current episode
        :param writer: tensorboard writer object
        :param done: if the episode is terminated (used for generalization of the training step function)
        :param td_steps: number of real experience steps used in the return estimation
        :return: None
    """
    global global_p_step
    global global_v_step

    def train_step(index, traj_len):
        """
        :param index: index of current transition within the trajectory
        :param traj_len: number of real experience steps used in the return estimation
        :return:
        """
        global global_p_step
        # state, action, reward are the current state, action, reward we look at
        # next_state, done are the end of trajectory next_state and done.
        state, action, reward, next_state, _done = states[index], actions[index], rewards[index], next_states[-1], \
                                                   dones[-1]

        ret_reward = (discount_factor ** np.arange(traj_len) * rewards[index:]).sum()
        # estimate of G_t
        estimated_discounted_return = ret_reward + discount_factor ** (traj_len) * \
                                      critic_net(next_state) * (1 - _done)

        # estimate of A_t
        advantage = estimated_discounted_return - critic_net(state)
        loss_history = critic_net.fit(state, estimated_discounted_return, batch_size=1, verbose=0)
        loss = loss_history.history['loss'][0]
        with writer.as_default():
            tf.summary.scalar('Loss_C', loss, step=global_v_step)
            tf.summary.scalar('Advantage', advantage[0][0], step=global_v_step)
        if mount_continuous:
            loss_history = policy.fit(state, np.concatenate((action[..., None], advantage), axis=-1), batch_size=1,
                                      verbose=0)
        else:
            loss_history = policy.fit(state, action, sample_weight=advantage, batch_size=1, verbose=0)

        loss = loss_history.history['loss'][0]
        with writer.as_default():
            tf.summary.scalar('Loss_A', loss, step=global_p_step)
        global_p_step += 1

    global_v_step = global_p_step
    if not done and len(episode_transitions) < td_steps:  # not enough steps to evaluate G_t by N-steps ahead
        return
    # unpack the last N-steps transitions into corresponding np.arrays
    states, actions, rewards, next_states, dones = tuple(map(np.asarray, zip(*episode_transitions[-td_steps:])))
    traj_len = len(episode_transitions[-td_steps:])
    if done:  # apply training steps for the remaining of the trajectory.
        for i in range(traj_len):
            train_step(i, traj_len - i)
    else:  # apply training step only for the current state we look at (at index 0).
        train_step(0, traj_len)


def train(td_steps=0, max_episodes=5000, exp_dir='def_dir', exp_name='No_Name', task=None, env=None, question='Q1',
          explore_start=False):
    """
    :param td_steps: determines the algorithm used.
                     td_steps=N>0 -> uses N-steps in the actor-critic method.
    :param max_episodes: max episodes to train
    :param exp_dir: directory for saving logs within the log-dir
    :param exp_name: name of the specific experiment
    :return:
    """

    train_fun = partial(actor_critic, td_steps=td_steps, task=task)

    path = os.path.join(base_log_path, exp_dir)
    n_actions = [3,  # acrobot
                 2,  # cartpole
                 action_size]  # mountain
    actions_map = np.array([[1 if i < n else 0 for i in range(action_size)] for n in n_actions])
    SOLVED_REWARDS = [-90, 475, 90]
    WORST_CASE = [-500, 0, -20]
    MAX_STEPS = [500, 500, 200]
    writer_train = tf.summary.create_file_writer(
        os.path.join(path, exp_name + '_' + datetime.datetime.now().strftime("%d%m%Y-%H%M")))
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = WORST_CASE[task]
    episode = -1
    for episode in range(max_episodes):
        state = env.reset()
        if episode < explore_start:
            low = env.env.low_state
            high = env.env.high_state
            rand_state = np.random.rand(len(high)) * (high - low) + low
            env.env.state = rand_state
        # task id is equal to the amount needed to pad from each side (i.e., 0 -acrobot , 1-cartpole, 2- mount)
        state = np.pad(state, task)
        state = state.reshape([1, state_size])
        episode_transitions = []
        score = 0
        actual_score = 0
        rs = 0
        for step in range(MAX_STEPS[task]):
            if task in [0, 1]:
                actions_distribution = tf.squeeze(policy(state))
                actions_distribution = actions_distribution * actions_map[task]
                action_rand = np.random.rand(1) * sum(actions_distribution)
                action = (np.cumsum(actions_distribution) < action_rand).numpy().sum()
                action_options = np.arange(action_size)

            else:
                if mount_continuous:
                    mu, sigma = policy(state)
                    action_cont = np.tanh(np.random.normal(mu, np.sqrt(sigma), 1))
                    histogram = np.tanh(np.random.normal(mu, np.sqrt(sigma), 1000))
                else:
                    actions_distribution = tf.squeeze(policy(state))
                    actions_distribution = actions_distribution * actions_map[task]
                    action_rand = np.random.rand(1) * sum(actions_distribution)
                    action = (np.cumsum(actions_distribution) < action_rand).numpy().sum()
                    # split [-1,1] to n_actions.
                    action_options = np.arange(-1.0, 1.0 + 1e-3, (1 - (-1)) / (action_sizes[task] - 1))
                    action_cont = [action_options[action]]

                with writer_train.as_default():
                    tf.summary.scalar('Velocity', state[0][task + 1], step=step + episode * (MAX_STEPS[task] + 50))
                    tf.summary.scalar('Location', state[0][task], step=step + episode * (MAX_STEPS[task] + 50))
                    tf.summary.scalar('Reward_Shaping', rs, step=step + episode * (MAX_STEPS[task] + 50))

            with writer_train.as_default():
                if not mount_continuous:
                    appearances = (actions_distribution.numpy() * 100).round()
                    histogram = []
                    for i in range(action_size):
                        histogram.extend([action_options[i] for j in range(int(appearances[i]))])
                tf.summary.histogram('Action_Dist', histogram, step=step + episode * (MAX_STEPS[task]))

            if task != 2:
                next_state, reward, done, _ = env.step(action)
            else:
                next_state, reward, done, _ = env.step(action_cont)
            score += reward
            episode_rewards[episode] += reward
            if task == 2:
                abs_accel = 100 * np.abs(
                    np.abs(next_state[1]) - np.abs(state[0][task + 1]))
                v_th = 1e-3
                direction = 1 if state[0][task] > -0.5 else -1
                if state[0][task + 1] * direction < v_th:
                    guidance_shaping = -action_cont[0] * direction
                else:
                    guidance_shaping = action_cont[0] * direction
                # reward += guidance_shaping  # *(np.abs(state[0][task] - (-0.5))) #+ np.abs(state[0][task+1])/0.07)
                with writer_train.as_default():
                    tf.summary.scalar('|Acceleration|', abs_accel, step=step + episode * (MAX_STEPS[task] + 50))
                    tf.summary.scalar('Guidance', guidance_shaping, step=step + episode * (MAX_STEPS[task] + 50))

            actual_score += reward
            next_state = np.pad(next_state, task)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()
            if not mount_continuous:
                action_one_hot = np.zeros((1, action_size))
                action_one_hot[0, action] = 1
            else:
                action_one_hot = action_cont
            transition = Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state,
                                    done=done)
            episode_transitions.append(transition)
            train_fun(episode_transitions, episode, writer=writer_train, done=done)
            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print(
                    "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                 round(average_rewards, 2)))
                if average_rewards > SOLVED_REWARDS[task] and episode > (explore_start+100):  # or (task == 2 and score > SOLVED_REWARDS[task]):
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                    checkpoint_directory = "training_checkpoints/"
                    checkpoint_dir_p = os.path.join(checkpoint_directory, question, taskID2name[task], exp_dir, 'actor')
                    checkpoint_dir_c = os.path.join(checkpoint_directory, question, taskID2name[task], exp_dir,
                                                    'critic')
                    checkpoint_dir = os.path.join(checkpoint_directory, question, taskID2name[task])
                    policy.save_weights(os.path.join(checkpoint_dir_p, 'policy_weights.ckpt'))
                    critic_net.save_weights(os.path.join(checkpoint_dir_c, 'critic_weights.ckpt'))
                with writer_train.as_default():
                    tf.summary.scalar('Score', score, step=episode)
                    tf.summary.scalar('Actual_Score', actual_score, step=episode)
                    tf.summary.scalar('Avg_Score', average_rewards, step=episode)
                break
            state = next_state
        if solved:
            break
    return episode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--exp', type=str, default='No_Name')
    parser.add_argument('--src', type=str, nargs='+', default=[None], choices=['cartpole', 'acrobot', 'mount', None])
    parser.add_argument('--dest', type=str, default='cartpole', choices=['cartpole', 'acrobot', 'mount'])
    parser.add_argument('--cont', default=False, action='store_true')
    parser.add_argument('--explore_start', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=5000)
    parser.add_argument('--Q', type=str, default='Q1', choices=['Q1', 'Q2', 'Q3'])
    args = parser.parse_args()
    return args


checkpoint_directories = {'cartpole': {'actor':  'training_checkpoints/Q1/cartpole/actor/',
                                       'critic': 'training_checkpoints/Q1/cartpole/critic/'},
                          'acrobot': {'actor':  'training_checkpoints/Q1/acrobot/actor/',
                                      'critic': 'training_checkpoints/Q1/acrobot/critic/'},
                          'mount': {'actor':  'training_checkpoints/Q1/mount/actor/',
                                    'critic': 'training_checkpoints/Q1/mount/critic/'}}

pre_trained_models = {}


def load_checkpoints(src):
    for s in src:
        actor = PolicyNetwork(state_size, action_sizes, task=taskName2ID[s], cont=mount_continuous,
                              name=f'policy_network_Q{len(src) + 1}_{s}')
        actor.load_weights(checkpoint_directories[s]['actor'] + 'policy_weights.ckpt')
        critic = CriticNetwork(state_size, discount_factor, task=taskName2ID[s],
                               name=f'critic_network_Q{len(src) + 1}_{s}')
        critic.load_weights(checkpoint_directories[s]['critic'] + 'critic_weights.ckpt')

        pre_trained_models[s] = {'actor': actor, 'critic': critic}


def check_validity(args):
    assert args.dest == 'mount' or (not args.cont and args.dest != 'mount')
    assert (args.Q == 'Q3' and len(args.src) == 2) or (args.Q != 'Q3' and len(args.src) == 1)
    assert args.N > 0
    assert args.n_episodes > 0
    assert args.explore_start >= 0


if __name__ == '__main__':
    args = parse_args()
    env_info = {'cartpole': ['CartPole-v1', 1], 'acrobot': ['Acrobot-v1', 0], 'mount': ['MountainCarContinuous-v0', 2]}
    task = env_info[args.dest][1]
    check_validity(args)
    mount_continuous = args.cont
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if mount_continuous:
        loss = GaussianActionLoss(alpha=0.0)
    else:
        loss = CategoricalCrossentropy(from_logits=True)
    if args.src[0] is not None:
        load_checkpoints(args.src)
    if args.Q == 'Q1':
        policy = PolicyNetwork(state_size, action_sizes, task=task, cont=mount_continuous)
        critic_net = CriticNetwork(state_size, discount_factor, task=task)
        policy.compile(optimizer=Adam(learning_rate_p), loss=loss)
        critic_net.compile(optimizer=Adam(learning_rate_c), loss=MeanSquaredError())
    elif args.Q == 'Q2':
        policy: PolicyNetwork = pre_trained_models[args.src[0]]['actor']
        critic_net: CriticNetwork = pre_trained_models[args.src[0]]['critic']
        policy.set_task(task)
        critic_net.set_task(task)
        policy.compile(optimizer=Adam(learning_rate_p), loss=loss)
        critic_net.compile(optimizer=Adam(learning_rate_c), loss=MeanSquaredError())
    elif args.Q == 'Q3':
        policy_list = []
        critic_list = []
        for s in args.src:
            pi = pre_trained_models[s]['actor']
            critic = pre_trained_models[s]['critic']
            policy_list.append(PN_PrevModel(state_size, pi))
            critic_list.append(PN_PrevModel(state_size, critic))

        policy: ProgressiveNetwork = ProgressiveNetwork(state_size, action_sizes, models=policy_list, task=task,
                                                        cont=mount_continuous,
                                                        name=f'progressive_network_Q3_{args.dest}')
        # critic_net = CriticNetwork(state_size, discount_factor, task, name=f'critic_network_Q3_{args.dest}')
        critic_net: ProgressiveCriticNetwork = ProgressiveCriticNetwork(state_size, models=critic_list,
                                                                        task=task,
                                                                        name=f'progressive_critic_network_Q3_{args.dest}')
        policy.compile(optimizer=Adam(learning_rate_p), loss=loss)
        critic_net.compile(optimizer=Adam(learning_rate_c), loss=MeanSquaredError())

    env = gym.make(env_info[args.dest][0])
    MAX_STEPS = [500, 500, 200]
    env._max_episode_steps = MAX_STEPS[env_info[args.dest][1]]

    start_at = time.time()
    n_episodes = train(td_steps=args.N, max_episodes=args.n_episodes, exp_dir=args.exp, exp_name=args.exp,
                       task=task, env=env, question=args.Q, explore_start=args.explore_start)
    # n_episodes = 10
    end_at = time.time()
    run_time = end_at - start_at  # time in seconds
    df = pd.DataFrame([[args.src, args.dest, run_time, n_episodes, args.exp]], columns=['Source', 'Dest', 'Time', 'Episodes','Exp'])
    if not os.path.exists('Output'):
        os.makedirs('Output')
    df.to_csv('Output/results.csv', index=False, header=not os.path.exists('Output/results.csv'), mode='a')
