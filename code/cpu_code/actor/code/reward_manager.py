import json


class RewardManager:
    def __init__(self):
        self.reward_weights = {}
        
        self.reward_money = 0.006
        self.reward_exp = 0.006
        self.reward_hp_point = 2.0
        self.reward_ep_rate = 0.75
        self.reward_kill = -0.6
        self.reward_dead = -1.0
        self.reward_tower_hp_point = 7.5
        self.reward_last_hit = 0.75
    
        
    def load_config(self, config_path):
        with open(config_path, mode='r') as f:
            self.reward_weights = json.load(f)
            
        for k, v in self.reward_weights.items():
            self.reward_weights[k] = float(v)
            
    def update_reward(self, episode):
        raise NotImplementedError
        
    
    