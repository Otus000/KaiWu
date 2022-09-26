from tensorflow.keras import layers, models, Input, optimizers, losses
# from tensorflow_probability.python.distributions import Normal
from collections import deque

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import copy

class SoftActorCritic:
    def __init__(self, state_shape, action_dim):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
        self.optimizer = optimizers.Adam(learning_rate=1e-3)
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.997
        self.temperature = 1.0

        policy_input = Input(shape=state_shape)
        x = layers.Dense(units=1024, activation='relu')(policy_input)
        x = layers.Dense(units=1024, activation='relu')(x)
        policy_mean = layers.Dense(units=action_dim, activation='linear')(x)
        log_policy_std = layers.Dense(units=action_dim, activation='linear')(x)
        log_policy_std_clipped = tf.clip_by_value(log_policy_std, -10, 2)
        self.policy_network = models.Model(inputs=policy_input, outputs=[policy_mean, log_policy_std_clipped])

        value_input = Input(shape=state_shape)
        x = layers.Dense(units=1024, activation='relu')(value_input)
        x = layers.Dense(units=1024, activation='relu')(x)
        value_output = layers.Dense(units=1, activation='linear')(x)
        self.value_network = models.Model(inputs=value_input, outputs=value_output)
        self.target_value_network = models.clone_model(self.value_network)
        self._update_target_value_network()

        Q_state_input = Input(shape=state_shape)
        Q_action_input = Input(shape=(action_dim))
        x = layers.concatenate([Q_state_input, Q_action_input])
        x = layers.Dense(units=1024, activation='relu')(x)
        x = layers.Dense(units=1024, activation='relu')(x)
        Q_output = layers.Dense(units=1, activation='linear')(x)
        self.Q_network_1 = models.Model(inputs=[Q_state_input, Q_action_input], outputs=Q_output)
        self.Q_network_2 = models.clone_model(self.Q_network_1)

    def _update_target_value_network(self):
        self.ema.apply(self.value_network.trainable_variables)
        for target_value_network_para, value_network_para in zip(self.target_value_network.trainable_variables, self.value_network.trainable_variables):
            target_value_network_para.assign(self.ema.average(value_network_para))

    def save_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def select_action(self, state):

        '''
        state = np.array([state])
        policy_mean, log_policy_std = self.policy_network(state)
        policy_mean = np.array(policy_mean[0])
        log_policy_std = np.array(log_policy_std[0])
        policy_std = np.exp(log_policy_std)

        # gaussian_distribution = Normal(policy_mean, policy_std)
        # action = np.tanh(gaussian_distribution.sample())

        np_normal = np.random.normal(policy_mean, policy_std)
        np_normal_tensor = tf.Tensor(np_normal, shape=(), dtype=float32)
        action = np.tanh(np_normal_tensor)
        '''
        # state = np.array([state])
        # policy_mean, log_policy_std = self.policy_network(state)
        action = [1,1,1,1,1,1]

        return action

    def update_weights(self, batch_size):
        batch_size = min(batch_size, len(self.replay_buffer))
        training_data = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        for data in training_data:
            s, a, r, n_s, d = data
            state.append(s)
            action.append(a)
            reward.append(r)
            next_state.append(n_s)
            done.append(d)
        state = np.array(state, dtype=np.float64)
        action = np.array(action, dtype=np.float64)
        reward = np.reshape(reward, newshape=(-1, 1))
        next_state = np.array(next_state, dtype=np.float64)
        done = np.reshape(done, newshape=(-1, 1))

        with tf.GradientTape() as tape:
            policy_mean, log_policy_std = self.policy_network(state)
            policy_std = tf.exp(log_policy_std)

            # gaussian_distribution = Normal(policy_mean, policy_std)
            # gaussian_sampling = tf.stop_gradient(gaussian_distribution.sample())
            np_normal = np.random.normal(policy_mean, policy_std)
            np_normal_tensor = tf.Tensor(np_normal, shape=(), dtype=float32)
            gaussian_sampling = tf.stop_gradient(np_normal_tensor)

            sample_action = tf.tanh(gaussian_sampling)
            logprob = gaussian_distribution.log_prob(gaussian_sampling) - tf.math.log(
                1.0 - tf.pow(sample_action, 2) + 1e-6)

            logprob = tf.reduce_mean(logprob, axis=-1, keepdims=True)
            new_Q_value = tf.math.minimum(tf.stop_gradient(self.Q_network_1([state, sample_action])), tf.stop_gradient(self.Q_network_2([state, sample_action])))
            advantage = tf.stop_gradient(logprob - new_Q_value)
            policy_loss = tf.reduce_mean(logprob * advantage)

        policy_network_grad = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(policy_network_grad, self.policy_network.trainable_variables))

        with tf.GradientTape() as tape:
            value = self.value_network(state)
            value_ = tf.stop_gradient(new_Q_value - self.temperature * logprob)
            value_loss = tf.reduce_mean(losses.mean_squared_error(value_, value))
        value_network_grad = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(value_network_grad, self.value_network.trainable_variables))

        target_value = tf.stop_gradient(self.target_value_network(next_state))
        Q_ = reward + self.gamma * (1 - done) * target_value

        with tf.GradientTape() as tape:
            Q_1 = self.Q_network_1([state, action])
            Q_1_loss = tf.reduce_mean(losses.mean_squared_error(Q_, Q_1))
        Q_network_1_grad = tape.gradient(Q_1_loss, self.Q_network_1.trainable_variables)
        self.optimizer.apply_gradients(zip(Q_network_1_grad, self.Q_network_1.trainable_variables))

        with tf.GradientTape() as tape:
            Q_2 = self.Q_network_2([state, action])
            Q_2_loss = tf.reduce_mean(losses.mean_squared_error(Q_, Q_2))
        Q_network_2_grad = tape.gradient(Q_2_loss, self.Q_network_2.trainable_variables)
        self.optimizer.apply_gradients(zip(Q_network_2_grad, self.Q_network_2.trainable_variables))

        self._update_target_value_network()

        return (
            np.array(Q_1_loss, dtype=np.float64),
            np.array(Q_2_loss, dtype=np.float64),
            np.array(policy_loss, dtype=np.float64),
            np.array(value_loss, dtype=np.float64)
        )

    def save_weights(self, path):
        self.policy_network.save_weights(path + '-policy_network.h5')
        self.value_network.save_weights(path + '-value_network.h5')
        self.Q_network_1.save_weights(path + '-Q_network_1.h5')
        self.Q_network_2.save_weights(path + '-Q_network_2.h5')

    def load_weights(self, path):
        self.policy_network.load_weights(path + '-policy_network.h5')
        self.value_network.load_weights(path + '-value_network.h5')
        self.Q_network_1.load_weights(path + '-Q_network_1.h5')
        self.Q_network_2.load_weights(path + '-Q_network_2.h5')