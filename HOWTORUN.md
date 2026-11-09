# Nine Men's Morris – MCTS Agent

This project implements a Monte Carlo Tree Search (MCTS) agent for the game of **Nine Men's Morris**. It includes UCT parameter exploration, configurable AI difficulty levels, and an interactive human-vs-AI mode.

## File structure

- `mcts.py` – Core MCTS implementation, including the node structure, UCB1 selection, rollout, and backpropagation.
- `test_part1.py` – Part 1: Adversarial testing (AI vs. random action chooser, AI vs. AI, and UCT different values expirements).
- `test_part2.py` – Part 2: AI difficulty tournament (Easy, Medium, and Hard) and computation-time calculations.
- `play_human.py` – Part 3: Interactive GUI mode for playing against the AI using the mouse.

## Prerequisites

Make sure you have **Python 3.9 or newer** installed.

Install the required libraries:

```bash
pip install famnit_gym@git+https://github.com/DomenSoberlFamnit/famnit-gym
```

## How to run

### 1. Adversarial search and UCT exploration (Part 1)

Runs headless matches to evaluate:

- MCTS performance against a random player.
- Head-to-head matches comparing different exploration constants:
  - `c = 0.2`
  - `c = 1.414`

Run:

```bash
python test-part1.py
```

### 2. Difficulty tournament and computation time (Part 2)

Runs a round-robin tournament between three AI difficulty levels:

| Difficulty | Iterations | Search depth | Blunder rate |
| ---------- | ---------: | -----------: | -----------: |
| Easy       |         25 |           10 |          20% |
| Medium     |        150 |           30 |           0% |
| Hard       |        400 |           50 |           0% |

The script also measures and prints the average thinking time per move for each difficulty level.

Run:

```bash
python test-part2.py
```

### 3. Human vs. AI interactive mode (Part 3)

Launches a graphical game window where a human can play against the AI using the mouse.

Run:

```bash
python play-human.py
```

#### Changing the AI difficulty

Open `play-human.py` and change the `AI_DIFFICULTY` variable:

```python
AI_DIFFICULTY = "Easy"  # Can choose between "Easy", "Medium" and "Hard"
```

#### Board controls

**Phase 1 – Placing pieces**

1. Click any unoccupied intersection to place a piece.
2. If you form a mill (three pieces in a row), click an opponent's piece to remove it.

**Phases 2 and 3 – Moving and flying**

1. Click one of your pieces to select it. The selected piece will be highlighted in green.
2. Click the destination intersection.
3. If you form a mill, click an opponent's piece to capture it.
