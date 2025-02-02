# Multi-Agent Reinforcement Learning (MARL) Framework

A PyTorch-based framework for training multiple agents in various environments using independent Q-learning. Supports both cooperative and competitive scenarios with dynamic observation/action space handling.

## 🎮 Supported Environments

| Environment | Type | Agents | Description |
|------------|------|--------|-------------|
| Simple Spread | Cooperative | 2-10 | Agents cover target landmarks while avoiding collisions |
| Simple Adversary | Mixed | 3-7 | Good agents cooperate against an adversary |
| Simple Tag | Competitive | 4-8 | Pursuit-evasion scenario with predator and prey |
| Knights Archers Zombies | Cooperative | 2-12 | Complex game with different agent types |
| StarCraft II | Mixed | 2 | Strategic real-time combat scenarios |

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd marl-framework
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 StarCraft II Setup

### Installation Steps

1. **Install StarCraft II**
   - Download [Battle.net](https://www.blizzard.com/download/)
   - Install StarCraft II through Battle.net
   - Complete the installation process

2. **Configure Environment**
```bash
# macOS
echo 'export SC2PATH="/Applications/StarCraft II"' >> ~/.zshrc
source ~/.zshrc

# Windows
set SC2PATH=C:\Program Files (x86)\StarCraft II

# Linux
export SC2PATH=~/StarCraft II
```

3. **Install Training Maps**
```bash
# Create Maps directory
mkdir -p "$SC2PATH/Maps"

# Download and extract maps
wget https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip
unzip mini_games.zip -d "$SC2PATH/Maps/"
```

### Available Maps
- **Simple64**: Basic 64x64 training map
- **AbyssalReef**: Professional 2-player map
- **Flat64**: Simplified 64x64 terrain map

### Running StarCraft II Training
```bash
# Test installation
python main.py --env starcraft --no-render --episodes 10

# Full training run
python main.py --env starcraft --episodes 2000
```

## 💻 Usage

### Interactive Mode
```bash
python main.py
```

### Command Line Mode
```bash
python main.py --env simple_spread --num-agents 3 --episodes 2000 --no-render
```

### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| --env | Environment name | (interactive) |
| --num-agents | Number of agents | (env default) |
| --no-render | Disable visualization | False |
| --episodes | Training episodes | 2000 |

## 🔧 Training Configuration

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | Agent learning speed |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.1 | Final exploration rate |
| Epsilon Decay | 0.995 | Exploration decay rate |
| Gamma | 0.95 | Reward discount factor |
| Batch Size | 32 | Training batch size |

### Neural Network Architecture
- Input Layer: Environment-specific observation size
- Hidden Layers: 2 x 128 units with ReLU
- Output Layer: Environment-specific action size
- Weight Initialization: Orthogonal
- Optimizer: Adam

## 📊 Performance Metrics

The framework tracks:
- Episode Total Reward
- Average Game Length
- Per-agent Performance
- Best Episode Score
- Training Progress

## 🏗️ Project Structure
```
marl-framework/
├── environments/        # Environment wrappers
├── agents/             # Neural network models
├── training/           # Training logic
└── main.py            # Entry point
```

## 🔍 Features
- Dynamic observation/action space handling
- Environment-specific configurations
- Real-time training visualization
- Flexible agent architectures
- Comprehensive metrics tracking
- Command-line and interactive modes

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
MIT License - see LICENSE file for details

