import json
import logging

import numpy as np


class RewardManager:
    def __init__(self):
        self.reward_weights_init = {
            "reward_money": 0.006,
            "reward_exp": 0.006,
            "reward_hp_point": 2.0,
            "reward_ep_rate": 0.75,
            "reward_kill": -0.6,
            "reward_dead": -1.0,
            "reward_tower_hp_point": 5.0,
            "reward_last_hit": 0.5,
        }

        self.reward_weights_begin = {
            "reward_money": 0.006,
            "reward_exp": 0.006,
            "reward_hp_point": 2.0,
            "reward_ep_rate": 0.75,
            "reward_kill": -0.8,  # -0.6
            "reward_dead": -1.5,  # -1.0
            "reward_tower_hp_point": 5.0,
            "reward_last_hit": 0.75,  # 0.5
        }

        self.reward_weights_middle = {
            "reward_money": 0.006,
            "reward_exp": 0.006,
            "reward_hp_point": 2.0,
            "reward_ep_rate": 0.75,
            "reward_kill": -0.6,
            "reward_dead": -1.2,
            "reward_tower_hp_point": 6.5,
            "reward_last_hit": 0.8,
        }

        self.reward_weights_end = {
            "reward_money": 0.006,
            "reward_exp": 0.006,
            "reward_hp_point": 2.0,
            "reward_ep_rate": 0.75,
            "reward_kill": -0.5,  # -0.6,
            "reward_dead": -1.2,  # -1.0,
            "reward_tower_hp_point": 8.0,  # 5.0,
            "reward_last_hit": 1.0,
        }
        self.reward_weights = {}
        self.decay_rate_init = 0.997
        self.decay_rate = 1
        self.reset()
        # self.reward_money = 0.008
        # self.reward_exp = 0.008
        # self.reward_hp_point = 2.0
        # self.reward_ep_rate = 0.8
        # self.reward_kill = -0.5
        # self.reward_dead = -1.0
        # self.reward_tower_hp_point = 10
        # self.reward_last_hit = 1

    def load_config(self, config_path):
        with open(config_path, mode='r') as f:
            self.reward_weights = json.load(f)

        for k, v in self.reward_weights.items():
            self.reward_weights[k] = float(v)

    def reset(self):
        self.reward_weights = self.reward_weights_init
        self.decay_rate = 1

    def update(self, states, frames):
        #self.change_stage(states)
        if (frames + 1) % 600 == 0:
            self.decay_rate *= self.decay_rate_init
            for k, v in self.reward_weights.items():
                if v > 0:
                    self.reward_weights[k] = self.decay_rate * self.reward_weights_init[k]

            # logging.info(f"DEBUG {frames}: current reward: {self.reward_weights}")

    def change_stage(self, states):
        if states[0] == 1:
            self.reward_weights = self.reward_weights_begin
        elif states[1] == 1:  # frame 2700
            self.reward_weights = self.reward_weights_middle
        elif states[2] == 1:  # frame 4500
            self.reward_weights = self.reward_weights_end

    def cal_multi_reward(self, reward):
        # return np.array([
        #     # total_reward
        #     # reward[-1],
        #     # reward_farming (exp, gold, mana)
        #     reward[2] * self.reward_exp + reward[-3] * self.reward_money + reward[
        #         1] * self.reward_ep_rate,
        #     # reward_kda (dead, kill, last_hit)
        #     reward[0] * self.reward_dead + reward[4] * self.reward_kill + reward[
        #         5] * self.reward_last_hit,
        #     # reward_damage (hp)
        #     reward[3] * self.reward_hp_point,
        #     # reward_pushing (tower_hp)
        #     reward[-2] * self.reward_tower_hp_point
        # ], dtype=np.float32)
        return np.array([
            # total_reward
            # reward[-1],
            # reward_farming (exp, gold, mana)
            reward[2] * self.reward_weights['reward_exp']
            + reward[-3] * self.reward_weights['reward_money']
            + reward[1] * self.reward_weights['reward_ep_rate'],
            # reward_kda (dead, kill, last_hit)
            reward[0] * self.reward_weights['reward_dead']
            + reward[4] * self.reward_weights['reward_kill']
            + reward[5] * self.reward_weights['reward_last_hit'],
            # reward_damage (hp)
            reward[3] * self.reward_weights['reward_hp_point'],
            # reward_pushing (tower_hp)
            reward[-2] * self.reward_weights['reward_tower_hp_point']
        ], dtype=np.float32)
