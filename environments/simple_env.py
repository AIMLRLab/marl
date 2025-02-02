import numpy as np
import logging
from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3, simple_tag_v3
from pettingzoo.butterfly import knights_archers_zombies_v10
from gymnasium.spaces import Box, Discrete

logger = logging.getLogger(__name__)

class MultiAgentEnv:
    """A wrapper for PettingZoo environments that standardizes the interface."""

    SUPPORTED_ENVS = {
        'simple_spread': simple_spread_v3,
        'simple_adversary': simple_adversary_v3,
        'simple_tag': simple_tag_v3,
        'knights_archers_zombies': knights_archers_zombies_v10
    }

    ENV_AGENT_COUNTS = {
        'simple_spread': (2, 10),  # min, max agents
        'simple_adversary': (3, 7),
        'simple_tag': (4, 8),  # Fixed number of agents
        'knights_archers_zombies': (2, 12)
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

    def __init__(self, env_name='simple_spread', num_agents=None, max_cycles=25, render_mode=None):
        """Initialize the environment.

        Args:
            env_name (str): Name of the environment to create
            num_agents (int, optional): Number of agents. Must be within env's limits
            max_cycles (int): Maximum steps per episode
            render_mode (str): Rendering mode ('human', 'rgb_array', or None)
        """
        if env_name not in self.SUPPORTED_ENVS:
            raise ValueError(f"Environment {env_name} not supported. Choose from: {list(self.SUPPORTED_ENVS.keys())}")

        # Build environment kwargs
        env_kwargs = {'max_cycles': max_cycles, 'render_mode': render_mode}

        # Create environment first to inspect its properties
        self.env = self.SUPPORTED_ENVS[env_name].parallel_env(**env_kwargs)
        self.possible_agents = self.env.possible_agents
        self.num_agents = len(self.possible_agents)

        # Get observation spaces for each agent
        self.state_dims = {}
        self.action_dims = {}
        self.process_fns = {}  # Add processing functions dictionary

        observations = self.env.reset()[0]
        for agent in self.possible_agents:
            obs = observations[agent]
            act_space = self.env.action_space(agent)

            # Get state dimension and processing function for this agent
            if isinstance(obs, np.ndarray):
                if len(obs.shape) > 1:
                    self.state_dims[agent] = int(np.prod(obs.shape))
                    self.process_fns[agent] = lambda x: x.reshape(-1)
                else:
                    self.state_dims[agent] = obs.shape[0]
                    self.process_fns[agent] = lambda x: x
            else:
                processed = np.array(obs).flatten()
                self.state_dims[agent] = len(processed)
                self.process_fns[agent] = lambda x: np.array(x).flatten()

            # Get action dimension for this agent
            if isinstance(act_space, Discrete):
                self.action_dims[agent] = act_space.n
            else:
                raise ValueError(f"Unsupported action space type for agent {agent}: {type(act_space)}")

            logger.info(f"Agent {agent} - Observation dim: {self.state_dims[agent]}, Action dim: {self.action_dims[agent]}")

    def reset(self):
        observations = self.env.reset()
        if isinstance(observations, tuple):
            observations = observations[0]
        return {
            agent: self.process_fns[agent](obs).astype(np.float32)
            for agent, obs in observations.items()
        }

    def step(self, actions):
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = {
            agent: terminations[agent] or truncations[agent]
            for agent in self.possible_agents
        }
        dones["__all__"] = all(dones.values())

        processed_obs = {
            agent: self.process_fns[agent](obs).astype(np.float32)
            for agent, obs in next_obs.items()
        }

        return processed_obs, rewards, dones, infos

    def render(self):
        """Render the environment if render_mode is set."""
        self.env.render()

    def close(self):
        self.env.close()
