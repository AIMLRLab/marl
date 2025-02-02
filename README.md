3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script with interactive environment selection:

```bash
python main.py
```

Or specify environment and parameters directly:

```bash
python main.py --env simple_spread --num-agents 3 --episodes 2000
```

### Command Line Arguments

- `--env`: Environment name (optional, interactive selection if not provided)
- `--num-agents`: Number of agents (optional, uses default if not provided)
- `--no-render`: Disable visualization
- `--episodes`: Number of training episodes (default: 2000)

## Project Structure

- `environments/`: Environment wrappers and configurations
- `agents/`: Neural network architectures and agent implementations
- `training/`: Training loops and metrics tracking

## Features

- Dynamic observation and action space handling
- Environment-specific agent configurations
- Real-time training metrics and visualization
- Support for both cooperative and competitive scenarios
- Configurable hyperparameters for training

## Training Parameters

- Learning rate: 0.001
- Epsilon decay: 0.995 (exploration rate)
- Gamma: 0.95 (discount factor)
- Batch size: 32
- Episodes: 2000 (default)

## Metrics

The framework tracks:

- Average total reward per episode
- Average game length
- Per-agent rewards
- Best performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
