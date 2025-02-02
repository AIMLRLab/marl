import numpy as np
import logging
from typing import Dict, Tuple
from gymnasium import spaces
import os
from absl import flags
from absl import app

# Patch for Python 3.11 compatibility
import random
old_shuffle = random.shuffle
random.shuffle = lambda x, *args: old_shuffle(x)

# Now import PySC2 after the patch
from pysc2.env import sc2_env
from pysc2.lib import features, actions

logger = logging.getLogger(__name__)

# Initialize flags only if they haven't been defined yet
FLAGS = flags.FLAGS
if 'sc2_run_config' not in FLAGS:
    flags.DEFINE_string("sc2_run_config", None, "Run configuration for StarCraft II")

# Make sure flags are parsed
try:
    flags.FLAGS.mark_as_parsed()
except:
    # Only pass empty args if flags haven't been parsed
    if not FLAGS.is_parsed():
        flags.FLAGS([''])  # Pass empty args list to avoid conflicts

class StarCraft2Wrapper:
    """StarCraft II environment wrapper."""

    AVAILABLE_MAPS = {
        'Simple64': "MoveToBeacon",
        'MoveToBeacon': "MoveToBeacon",
        'CollectMineralShards': "CollectMineralShards",
        'DefeatRoaches': "DefeatRoaches"
    }

    def __init__(self, map_name: str = "MoveToBeacon", render_mode: str = None):
        """Initialize StarCraft II environment."""
        sc2_path = os.getenv('SC2PATH')
        if not sc2_path:
            raise ValueError("SC2PATH environment variable not set.")

        logger.info(f"Using StarCraft II from: {sc2_path}")
        logger.info(f"Loading map: {self.AVAILABLE_MAPS[map_name]}")

        # PettingZoo compatibility attributes
        self.possible_agents = ["agent_0"]
        self.agent_selection = self.possible_agents[0]
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        try:
            self.env = sc2_env.SC2Env(
                map_name=self.AVAILABLE_MAPS[map_name],
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True
                ),
                step_mul=8,
                visualize=render_mode is not None,
                game_steps_per_episode=150
            )
            self.action_space = spaces.Discrete(7056)  # 84x84 grid
            self.observation_space = spaces.Box(low=0, high=1, shape=(7056,), dtype=np.float32)
            self.timestep = None

        except Exception as e:
            logger.error(f"Failed to initialize SC2 environment: {str(e)}")
            raise

    def step(self, action):
        """Execute action and return new state."""
        try:
            # Convert flattened action back to x,y coordinates
            x = action // 84
            y = action % 84

            # First, check what actions are available
            available_actions = self.timestep.observation.available_actions

            # Select army if we haven't yet and it's available
            if actions.FUNCTIONS.select_army.id in available_actions:
                select_action = actions.FUNCTIONS.select_army("select")
                self.timestep = self.env.step([select_action])[0]
                available_actions = self.timestep.observation.available_actions

            # Move if it's available, otherwise no-op
            if actions.FUNCTIONS.Move_screen.id in available_actions:
                move_action = actions.FUNCTIONS.Move_screen("now", [x, y])
                self.timestep = self.env.step([move_action])[0]
            else:
                no_op = actions.FUNCTIONS.no_op()
                self.timestep = self.env.step([no_op])[0]

            # Process results
            obs = self._process_observation(self.timestep)
            reward = float(self.timestep.reward)
            done = self.timestep.last()

            # Update PettingZoo attributes
            self.rewards[self.agent_selection] = reward
            self.terminations[self.agent_selection] = done
            self.truncations[self.agent_selection] = False

            return obs, self.rewards, self.terminations, self.infos

        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            raise

    def _process_observation(self, timestep):
        """Convert SC2 observation to normalized array."""
        # Get screen features
        screen = timestep.observation.feature_screen
        # Take player_relative layer to see where units are
        screen = screen.player_relative
        return screen.reshape(-1).astype(np.float32) / 4.0  # Normalize by max value

    def reset(self):
        """Reset the environment."""
        self.timestep = self.env.reset()[0]
        obs = self._process_observation(self.timestep)

        # Reset PettingZoo attributes
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        return obs

    def observe(self, agent):
        """Get observation for specific agent."""
        if agent not in self.possible_agents:
            raise ValueError(f"Invalid agent: {agent}")
        return self._process_observation(self.timestep)

    def close(self):
        """Close the environment."""
        if hasattr(self, 'env'):
            self.env.close()
