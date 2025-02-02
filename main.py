import logging
import torch
import numpy as np
from environments.simple_env import MultiAgentEnv
from agents.independent_q import QNetwork
from training.trainer import GameMetrics
import argparse
import inquirer

# Configure logging to be more concise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def train(env, agents, metrics, episodes=2000,
          epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,  # Slower decay, higher final exploration
          learning_rate=0.001,  # Lower learning rate for stability
          gamma=0.95,  # Lower gamma to focus more on immediate rewards
          batch_size=32):
    logger.info(f"Training {len(agents)} agents | ε={epsilon_start:.2f}->{epsilon_end} | lr={learning_rate}")
    logger.info(f"Render mode: {env.render_mode}")

    optimizers = {
        agent: torch.optim.Adam(agents[agent].parameters(), lr=learning_rate)
        for agent in agents
    }

    epsilon = epsilon_start
    best_reward = float('-inf')
    best_episode = 0

    for episode in range(episodes):
        observations = env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        done = {"__all__": False}
        steps = 0

        while not done["__all__"]:
            # Always try to render at the start of each step
            env.render()

            actions = {}
            for agent in env.possible_agents:
                if np.random.random() < epsilon:
                    # Random action
                    actions[agent] = np.random.randint(env.action_dims[agent])
                else:
                    # Greedy action
                    state = torch.FloatTensor(observations[agent]).unsqueeze(0)
                    with torch.no_grad():
                        q_values = agents[agent](state)
                    actions[agent] = q_values.argmax().item()

            next_observations, rewards, done, _ = env.step(actions)
            steps += 1

            # Normalize and clip rewards
            normalized_rewards = {
                agent: max(min(reward / 20.0, 1.0), -1.0)  # Clip between -1 and 1
                for agent, reward in rewards.items()
            }

            # Update agents
            for agent in env.possible_agents:
                optimizer = optimizers[agent]
                optimizer.zero_grad()

                current_obs = torch.FloatTensor(observations[agent])
                next_obs = torch.FloatTensor(next_observations[agent])

                current_q = agents[agent](current_obs)[actions[agent]]
                with torch.no_grad():
                    next_q = agents[agent](next_obs)
                    target = normalized_rewards[agent] + gamma * next_q.max() * (not done[agent])

                loss = (current_q - target) ** 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agents[agent].parameters(), 0.5)  # More aggressive clipping
                optimizer.step()

                episode_rewards[agent] += rewards[agent]  # Keep original rewards for logging

            observations = next_observations

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        metrics.update(episode_rewards, done)

        if (episode + 1) % 100 == 0:
            stats = metrics.get_stats()
            avg_reward = stats["avg_total_reward"]
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_episode = episode + 1
                logger.info(f"Episode {episode+1:4d} | Reward: {avg_reward:7.1f} (best) | ε: {epsilon:.2f} | Steps: {steps}")
            else:
                logger.info(f"Episode {episode+1:4d} | Reward: {avg_reward:7.1f} | ε: {epsilon:.2f} | Best: {best_reward:.1f} @ {best_episode}")

def get_env_selection():
    environments = [
        'simple_spread',
        'simple_adversary',
        'simple_tag',
        'knights_archers_zombies'
    ]

    questions = [
        inquirer.List('env',
                     message="Which environment would you like to train on?",
                     choices=environments),
        inquirer.Text('num_agents',
                     message="How many agents? (press enter for default)")
    ]

    answers = inquirer.prompt(questions)

    # Convert num_agents to int if provided
    if answers['num_agents'].strip():
        try:
            answers['num_agents'] = int(answers['num_agents'])
        except ValueError:
            answers['num_agents'] = None
    else:
        answers['num_agents'] = None

    return answers

def main():
    parser = argparse.ArgumentParser(description='Train MARL agents')
    parser.add_argument('--env', type=str, help='Environment to train in')
    parser.add_argument('--num-agents', type=int, help='Number of agents (must be within env limits)')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes to train')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Set up logging based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # If env not specified via command line, use interactive selection
    if not args.env:
        answers = get_env_selection()
        args.env = answers['env']
        if not args.num_agents:
            args.num_agents = answers['num_agents']

    # Initialize environment with optional rendering
    render_mode = None if args.no_render else "human"
    env = MultiAgentEnv(
        env_name=args.env,
        num_agents=args.num_agents,
        max_cycles=25,
        render_mode=render_mode,
        debug=args.debug
    )

    # Initialize agents with correct dimensions for each agent
    agents = {
        agent: QNetwork(env.state_dims[agent], env.action_dims[agent])
        for agent in env.possible_agents
    }
    metrics = GameMetrics()
    train(env, agents, metrics, episodes=args.episodes)
    env.close()

if __name__ == "__main__":
    main()
