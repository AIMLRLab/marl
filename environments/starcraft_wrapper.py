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
                    feature_dimensions=features.Dimensions(screen=32, minimap=16),
                    use_feature_units=True,
                    hide_specific_actions=True
                ),
                step_mul=4,  # Slower steps (was 8)
                visualize=render_mode is not None,
                game_steps_per_episode=400  # Longer episodes (was 200)
            )

            # Smaller observation space
            self.observation_space = spaces.Box(
                low=-1, high=1,
                shape=(5,),  # [marine_x, marine_y, beacon_x, beacon_y, distance]
                dtype=np.float32
            )

            # Simpler action space (8 directions + no-op)
            self.action_space = spaces.Discrete(9)

            # Metrics
            self.episode_rewards = []
            self.episode_lengths = []
            self.min_distances = []

            self.timestep = None

        except Exception as e:
            logger.error(f"Failed to initialize SC2 environment: {str(e)}")
            raise

    def step(self, action):
        """Execute action and return new state."""
        try:
            # Get current state before action
            current_obs = self._process_observation(self.timestep)
            old_distance = current_obs[4]

            # Execute movement
            if actions.FUNCTIONS.Move_screen.id in self.timestep.observation.available_actions:
                # Convert normalized coordinates to screen space
                marine_x = int((current_obs[0] + 1) * 16)
                marine_y = int((current_obs[1] + 1) * 16)

                # Simple action to movement conversion
                actions_map = {
                    0: (0, 0),    # no-op
                    1: (0, 1),    # up
                    2: (1, 1),    # up-right
                    3: (1, 0),    # right
                    4: (1, -1),   # down-right
                    5: (0, -1),   # down
                    6: (-1, -1),  # down-left
                    7: (-1, 0),   # left
                    8: (-1, 1),   # up-left
                }

                dx, dy = actions_map[action]

                # Fixed step size for consistent learning
                target_x = min(max(marine_x + dx * 3, 0), 31)
                target_y = min(max(marine_y + dy * 3, 0), 31)

                self.timestep = self.env.step([actions.FUNCTIONS.Move_screen("now", [target_x, target_y])])[0]
            else:
                self.timestep = self.env.step([actions.FUNCTIONS.select_army("select")])[0]

            # Get new state
            new_obs = self._process_observation(self.timestep)
            new_distance = new_obs[4]

            # Enhanced reward structure for better learning
            reward = 0.0

            # Distance improvement reward (scaled for learning)
            distance_delta = old_distance - new_distance
            if distance_delta > 0:
                # Reward for moving closer (more reward when closer to beacon)
                reward += distance_delta * (1.0 / (new_distance + 0.1))
            else:
                # Small penalty for moving away
                reward += distance_delta * 0.5

            # Sparse reward for reaching beacon
            if new_distance < 0.1:
                reward += 5.0
                logger.info(f"Success! Distance: {new_distance:.3f}")

            # Small penalty for no-op to encourage exploration
            if action == 0:
                reward -= 0.01

            done = self.timestep.last() or new_distance < 0.1  # End episode on success

            # Log metrics for analysis
            if done:
                self.episode_rewards.append(reward)
                self.min_distances.append(new_distance)
                logger.info(
                    f"Episode finished - "
                    f"Distance: {new_distance:.2f}, "
                    f"Total reward: {reward:.2f}, "
                    f"Min distance: {min(self.min_distances):.2f}, "
                    f"Distance delta: {distance_delta:.3f}"
                )

            return new_obs, reward, done, {}

        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            raise

    def _process_observation(self, timestep):
        """Convert SC2 observation to normalized array."""
        try:
            screen = timestep.observation.feature_screen
            player_relative = screen.player_relative

            # Find marine and beacon with fallback values
            marine_coords = np.where(player_relative == features.PlayerRelative.SELF)
            beacon_coords = np.where(player_relative == features.PlayerRelative.NEUTRAL)

            # Default positions if units not found
            marine_x, marine_y = 0.0, 0.0
            beacon_x, beacon_y = 0.0, 0.0

            if len(marine_coords[0]) > 0 and len(marine_coords[1]) > 0:
                marine_y, marine_x = np.mean(marine_coords[0]), np.mean(marine_coords[1])
                marine_x = (marine_x / 32.0) * 2 - 1
                marine_y = (marine_y / 32.0) * 2 - 1

            if len(beacon_coords[0]) > 0 and len(beacon_coords[1]) > 0:
                beacon_y, beacon_x = np.mean(beacon_coords[0]), np.mean(beacon_coords[1])
                beacon_x = (beacon_x / 32.0) * 2 - 1
                beacon_y = (beacon_y / 32.0) * 2 - 1

            # Calculate normalized distance (with safety check)
            distance = np.sqrt((marine_x - beacon_x)**2 + (marine_y - beacon_y)**2) / np.sqrt(2)

            obs = np.array([
                marine_x, marine_y,
                beacon_x, beacon_y,
                distance
            ], dtype=np.float32)

            logger.debug(f"Marine at ({marine_x:.2f}, {marine_y:.2f}), "
                        f"Beacon at ({beacon_x:.2f}, {beacon_y:.2f}), "
                        f"Distance: {distance:.2f}")

            return obs

        except Exception as e:
            logger.error(f"Error in observation processing: {str(e)}")
            # Return safe default observation
            return np.zeros(5, dtype=np.float32)

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
