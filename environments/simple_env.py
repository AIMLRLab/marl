import sys
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Debug environment
logger.debug("=== Environment Debug Info ===")
logger.debug(f"Python executable: {sys.executable}")
logger.debug(f"Python version: {sys.version}")
logger.debug(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
logger.debug(f"Virtual env: {os.getenv('VIRTUAL_ENV')}")
logger.debug("=== Python Path ===")
for path in sys.path:
    logger.debug(f"Path: {path}")

# Debug package installations
logger.debug("=== Package Installation Check ===")
try:
    import pkg_resources
    installed_packages = [d for d in pkg_resources.working_set]
    logger.debug("Installed packages:")
    for package in installed_packages:
        logger.debug(f"  {package}")
except Exception as e:
    logger.error(f"Failed to list packages: {e}")

# Debug specific imports
logger.debug("=== Import Attempts ===")
for package in ['pygame', 'pysc2', 'pettingzoo']:
    try:
        __import__(package)
        logger.debug(f"Successfully imported {package}")
    except ImportError as e:
        logger.error(f"Failed to import {package}: {e}")
        if hasattr(e, '__traceback__'):
            logger.error(f"Traceback for {package}:", exc_info=True)

logger.debug("Attempting to import pettingzoo.mpe...")
try:
    from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3, simple_tag_v3
    logger.debug("Successfully imported pettingzoo.mpe")
except ImportError as e:
    logger.error(f"Failed to import pettingzoo.mpe: {e}")
    logger.error(f"Looking for pettingzoo in: {[p for p in sys.path if 'pettingzoo' in p]}")

logger.debug("Attempting to import pettingzoo.classic...")
try:
    from pettingzoo.classic import connect_four_v3, tictactoe_v3, chess_v6, rps_v2, go_v5
    logger.debug("Successfully imported pettingzoo.classic")
except ImportError as e:
    logger.error(f"Failed to import pettingzoo.classic: {e}")

logger.debug("Attempting to import pettingzoo.butterfly...")
try:
    from pettingzoo.butterfly import knights_archers_zombies_v10, pistonball_v6
    logger.debug("Successfully imported pettingzoo.butterfly")
except ImportError as e:
    logger.error(f"Failed to import pettingzoo.butterfly: {e}")

from gymnasium.spaces import Box, Discrete
from environments.starcraft_wrapper import StarCraft2Wrapper
from typing import Optional, Dict

class MultiAgentEnv:
    """A wrapper for PettingZoo environments that standardizes the interface."""

    SUPPORTED_ENVS = {
        # MPE (Multi-Particle Environments)
        'simple_spread': simple_spread_v3,
        'simple_adversary': simple_adversary_v3,
        'simple_tag': simple_tag_v3,

        # Classic Games
        'connect_four': connect_four_v3,
        'tictactoe': tictactoe_v3,
        'chess': chess_v6,
        'rps': rps_v2,  # Rock, Paper, Scissors
        'go': go_v5,

        # Complex Games
        'knights_archers_zombies': knights_archers_zombies_v10,
        'pistonball': pistonball_v6,

        # StarCraft II
        'starcraft': StarCraft2Wrapper,
    }

    ENV_AGENT_COUNTS = {
        # MPE
        'simple_spread': (2, 10),
        'simple_adversary': (3, 7),
        'simple_tag': (4, 8),

        # Classic
        'connect_four': (2, 2),
        'tictactoe': (2, 2),
        'chess': (2, 2),
        'rps': (2, 2),
        'go': (2, 2),

        # Complex
        'knights_archers_zombies': (2, 12),
        'pistonball': (2, 20),

        # StarCraft II
        'starcraft': (2, 8),  # Supports 2-8 agents for different scenarios
    }

    # Define environment-specific configurations
    ENV_CONFIGS = {
        'simple_spread': {
            'param': 'N'
        },
        'simple_adversary': {
            'param': 'N_good',  # Changed from num_good to N_good
            'fixed': {'N_adversaries': 1}  # Always 1 adversary
        },
        'knights_archers_zombies': {
            'split': True,  # Indicates this env needs agent count split
            'params': {'num_knights': 0.5, 'num_archers': 0.5}  # Split ratio
        }
    }

    def __init__(
        self,
        env_name: str,
        num_agents: Optional[int] = None,
        max_cycles: int = 25,
        render_mode: Optional[str] = None,
        debug: bool = False,
        map_name: str = "Simple64"  # Add this parameter
    ):
        """Initialize environment wrapper.

        Args:
            env_name: Name of environment to load
            num_agents: Number of agents (must be within env limits)
            max_cycles: Maximum steps per episode
            render_mode: Rendering mode (human or None)
            debug: Enable debug logging
            map_name: Map name for StarCraft II environment
        """
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Keep original render_mode request
        self.render_mode = render_mode
        self.is_parallel = False
        logger.debug(f"Requested render mode: {render_mode}")

        try:
            env_class = self.SUPPORTED_ENVS[env_name]
            logger.debug(f"Initializing {env_name} environment")

            if env_name == 'starcraft':
                self.env = env_class(map_name=map_name, render_mode=render_mode)
                self.is_parallel = False
            elif env_name in ['simple_spread', 'simple_adversary', 'simple_tag']:
                self.env = env_class.parallel_env(max_cycles=max_cycles, render_mode=render_mode)
                self.is_parallel = True
            else:
                self.env = env_class.env(render_mode=render_mode)
                self.is_parallel = False

        except Exception as e:
            logger.error(f"Failed to initialize environment {env_name}: {str(e)}")
            raise

        self.env_name = env_name
        self.possible_agents = self.env.possible_agents
        self.state_dims = {}
        self.action_dims = {}
        self.process_fns = {}

        # Initialize spaces
        self._init_spaces()

        logger.info(f"Environment {env_name} initialized with {len(self.possible_agents)} agents")
        for agent in self.possible_agents:
            logger.info(f"Agent {agent} - Observation dim: {self.state_dims[agent]}, Action dim: {self.action_dims[agent]}")

    def _process_observation(self, observation):
        """Process different types of observations into flat numpy arrays."""
        logger.debug(f"Processing observation type: {type(observation)}")
        logger.debug(f"Raw observation: {observation}")

        if isinstance(observation, dict):
            logger.debug(f"Dict observation keys: {observation.keys()}")

            # Handle dictionary observations (like tictactoe)
            if 'observation' in observation:
                obs = observation['observation']
            elif 'board' in observation:
                obs = observation['board']
            else:
                # For tictactoe, concatenate board and mask
                board = np.array(observation.get('board', [])).flatten()
                mask = np.array(observation.get('action_mask', [])).flatten()
                obs = np.concatenate([board, mask])
                logger.debug(f"Board shape: {board.shape}, Mask shape: {mask.shape}")
        elif isinstance(observation, (int, float)):
            obs = np.array([observation])
        elif isinstance(observation, np.ndarray):
            obs = observation.flatten()
        else:
            obs = np.array(observation).flatten()

        # Ensure obs is 1D and float32
        obs = obs.reshape(-1).astype(np.float32)
        logger.debug(f"Final observation shape: {obs.shape}")
        return obs

    def _get_state_dim(self, observation):
        """Get the flattened dimension of an observation."""
        try:
            processed = self._process_observation(observation)
            dim = len(processed)
            logger.debug(f"State dimension calculated: {dim}")
            return dim
        except Exception as e:
            logger.error(f"Failed to process observation: {observation}")
            logger.error(f"Observation type: {type(observation)}")
            logger.error(f"Error: {str(e)}")
            raise

    def _get_action_dim(self, act_space):
        """Extract action dimension from different space types."""
        # Handle method case (classic games sometimes return methods)
        if callable(act_space):
            try:
                # Try with current agent if it's a method requiring agent parameter
                act_space = act_space(self.current_agent)
            except TypeError:
                # If that fails, try without parameters
                act_space = act_space()

        if isinstance(act_space, Discrete):
            return act_space.n
        elif isinstance(act_space, dict):  # For Dict spaces
            if 'action_mask' in act_space:  # Games like Tictactoe
                return len(act_space['action_mask'])
            elif 'move' in act_space:  # Games like Chess
                return act_space['move'].n
        elif hasattr(act_space, 'n'):  # Fallback for other discrete-like spaces
            return act_space.n

        raise ValueError(f"Unsupported action space type: {type(act_space)}")

    def _process_action_space(self, agent, act_space):
        """Process action space for an agent."""
        try:
            self.current_agent = agent  # Store current agent for _get_action_dim
            self.action_dims[agent] = self._get_action_dim(act_space)
            logger.debug(f"Processed action space for {agent}: dim={self.action_dims[agent]}")
        except ValueError as e:
            logger.error(f"Failed to process action space for {agent}: {act_space}")
            logger.error(f"Action space type: {type(act_space)}")
            raise e

    def reset(self):
        """Reset the environment and return initial observations."""
        if self.is_parallel:
            observations = self.env.reset()[0]
        else:
            self.env.reset()
            observations = {agent: self.env.observe(agent) for agent in self.possible_agents}

        processed_obs = {}
        for agent, obs in observations.items():
            processed = self._process_observation(obs)
            logger.debug(f"Reset: Agent {agent} observation shape: {processed.shape}")
            processed_obs[agent] = processed

        return processed_obs

    def step(self, actions):
        """Execute actions in the environment."""
        if self.is_parallel:
            observations, rewards, terminations, truncations, _ = self.env.step(actions)
            done = {
                agent: terminations[agent] or truncations[agent]
                for agent in self.possible_agents
            }
            done["__all__"] = all(done.values())
        else:
            observations = {}
            rewards = {}
            done = {}

            for agent in self.possible_agents:
                # Skip if agent is already done
                if agent in self.env.terminations and self.env.terminations[agent]:
                    observations[agent] = self.env.observe(agent)
                    rewards[agent] = 0.0
                    done[agent] = True
                    continue

                # Get action mask for current agent
                obs = self.env.observe(agent)
                action = actions.get(agent)  # Get action safely

                if isinstance(obs, dict) and 'action_mask' in obs:
                    if not obs['action_mask'][action]:
                        legal_moves = np.where(obs['action_mask'])[0]
                        if len(legal_moves) > 0:
                            action = np.random.choice(legal_moves)
                            actions[agent] = action

                # Execute step
                try:
                    if action is not None:  # Only step if we have a valid action
                        self.env.step(action)
                except ValueError as e:
                    if "dead" in str(e):
                        observations[agent] = self.env.observe(agent)
                        rewards[agent] = 0.0
                        done[agent] = True
                        continue
                    raise e

                # Get updated state
                observations[agent] = self.env.observe(agent)
                rewards[agent] = float(self.env.rewards.get(agent, 0.0))
                done[agent] = self.env.terminations[agent] or self.env.truncations[agent]

            done["__all__"] = all(done.values())

        processed_obs = {
            agent: self._process_observation(obs).astype(np.float32)
            for agent, obs in observations.items()
        }

        logger.debug(f"Step complete - Observations: {[obs.shape for obs in processed_obs.values()]}")
        return processed_obs, rewards, done, {}

    def render(self):
        """Render the environment if render_mode is set."""
        if self.render_mode == "human":
            try:
                if hasattr(self.env, 'render'):
                    self.env.render()
                elif hasattr(self.env, 'aec_env') and hasattr(self.env.aec_env, 'render'):
                    self.env.aec_env.render()
            except Exception as e:
                logger.warning(f"Failed to render: {str(e)}")

    def close(self):
        """Close the environment and any displays."""
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
            if hasattr(self.env, 'render_mode') and self.env.render_mode == "human":
                import pygame
                pygame.display.quit()
                pygame.quit()
        except Exception as e:
            logger.warning(f"Error during environment cleanup: {str(e)}")

    def _init_spaces(self):
        """Initialize observation and action spaces for all agents."""
        logger.debug("Initializing spaces for all agents")

        # Initialize spaces based on environment type
        if self.is_parallel:
            observations = self.env.reset()[0]
            for agent in self.possible_agents:
                logger.debug(f"Processing parallel agent: {agent}")
                obs = observations[agent]
                act_space = self.env.action_space(agent)

                # Process observation space
                self.state_dims[agent] = self._get_state_dim(obs)
                self.process_fns[agent] = self._process_observation

                # Process action space
                self._process_action_space(agent, act_space)
        else:
            # For classic games
            self.env.reset()
            for agent in self.possible_agents:
                logger.debug(f"Processing classic game agent: {agent}")
                observation = self.env.observe(agent)

                # Process observation space
                self.state_dims[agent] = self._get_state_dim(observation)
                self.process_fns[agent] = self._process_observation

                # Get action space based on environment type
                if hasattr(self.env, 'action_space'):
                    act_space = self.env.action_space
                elif hasattr(self.env, 'action_spaces'):
                    act_space = self.env.action_spaces[agent]
                else:
                    act_space = self.env.observation_space(agent)

                logger.debug(f"Action space for {agent}: {type(act_space)}")
                self._process_action_space(agent, act_space)
