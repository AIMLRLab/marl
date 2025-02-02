import numpy as np
import logging
from typing import Dict, Tuple
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import features, actions
import os
from absl import flags
from absl import app

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
        'Simple64': "CollectMineralShards",
        'CollectMineralShards': "CollectMineralShards",  # Simplest map - collect minerals
        'MoveToBeacon': "MoveToBeacon",  # Easiest map - just move to a beacon
        'DefeatRoaches': "DefeatRoaches"  # Combat map - defeat enemy units
    }

    def __init__(self, map_name: str = "MoveToBeacon", render_mode: str = None):
        """Initialize StarCraft II environment."""
        # Verify SC2PATH environment variable
        sc2_path = os.getenv('SC2PATH')
        if not sc2_path:
            raise ValueError("SC2PATH environment variable not set. Please follow README instructions.")

        logger.info(f"Using StarCraft II from: {sc2_path}")
        logger.info(f"Attempting to load map: {self.AVAILABLE_MAPS[map_name]}")

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(actions.FUNCTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(7056,), dtype=np.float32)

        try:
            self.env = sc2_env.SC2Env(
                map_name=self.AVAILABLE_MAPS[map_name],
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True
                ),
                step_mul=8,
                visualize=render_mode is not None
            )
        except Exception as e:
            logger.error(f"Failed to initialize SC2 environment: {str(e)}")
            raise

        # Rest of the initialization code remains the same
        self.map_name = map_name
        self.n_agents = 1
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.state_dims = {agent: 7056 for agent in self.possible_agents}
        self.action_dims = {agent: len(actions.FUNCTIONS) for agent in self.possible_agents}

        # Initialize state variables
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._step_count = 0
        self.max_steps = 1000

        self.render_mode = render_mode
        logger.info(f"Initialized StarCraft II environment: {map_name}")

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment."""
        self._step_count = 0
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}

        self.obs = self.env.reset()[0]
        return self._process_observation(self.obs.observation)

    def step(self, action):
        """Execute action and return new state."""
        try:
            # Convert action to SC2 action
            sc2_action = self._process_action(action)
            timestep = self.env.step([sc2_action])[0]

            # Process observation and rewards
            obs = self._process_observation(timestep.observation)
            reward = float(timestep.reward)
            done = timestep.last()
            info = {}

            return obs, reward, done, info

        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            raise

    def _process_action(self, action):
        """Convert normalized action to SC2 action."""
        # For MoveToBeacon, we just need to move to the beacon
        try:
            # Select all marines
            if self._can_do(actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

            # Move to beacon
            if self._can_do(actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", action)

            # No-op if no other action is available
            return actions.FUNCTIONS.no_op()

        except Exception as e:
            logger.error(f"Error processing action: {str(e)}")
            return actions.FUNCTIONS.no_op()

    def _can_do(self, action_id):
        """Check if an action is available."""
        return action_id in self.obs.observation.available_actions

    def _process_observation(self, obs):
        """Convert SC2 observation to normalized array."""
        # Flatten and normalize the screen features
        screen = obs.feature_screen
        flat_screen = screen.reshape(-1)
        return np.clip(flat_screen / 255.0, 0, 1)

    def observe(self, agent: str) -> np.ndarray:
        """Get observation for specific agent."""
        if agent not in self.possible_agents:
            raise ValueError(f"Invalid agent: {agent}")

        return np.random.uniform(0, 1, self.state_dims[agent]).astype(np.float32)

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode == "human":
            logger.info("Rendering not implemented for simplified wrapper")

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self, 'env'):
            self.env.close()
