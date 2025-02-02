import logging
import numpy as np
import torch

# Using GameMetrics from ppo-agents.py
"""Reference: code/ppo-agents.py lines 37-59"""
class GameMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.game_lengths = []
        self.agent_rewards = {}

    def update(self, rewards, done):
        if done["__all__"]:
            total_reward = sum(rewards.values())
            self.episode_rewards.append(total_reward)
            self.game_lengths.append(len(rewards))

            # Track per-agent rewards
            for agent, reward in rewards.items():
                if agent not in self.agent_rewards:
                    self.agent_rewards[agent] = []
                self.agent_rewards[agent].append(reward)

    def get_stats(self):
        return {
            "avg_total_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_game_length": np.mean(self.game_lengths) if self.game_lengths else 0,
            "avg_agent_rewards": {
                agent: np.mean(rewards) if rewards else 0
                for agent, rewards in self.agent_rewards.items()
            }
        }
