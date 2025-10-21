import functools
import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector as AgentSelector

from famnit_gym.envs.mill.mill_model import MillModel

# Create the Mill environment.
def env(render_mode=None):
    internal_render_mode = None if render_mode != "human" else render_mode
    env = MillEnv(render_mode=render_mode)
    return env

# Return the Mill transition model for off-line computations.
def transition_model(env):
    if type(env) is not MillEnv:
        raise AttributeError(f'The environment must be an instance of the MillEnv class.')
    return env._model.clone()

class MillEnv(AECEnv):
    metadata = {
        "name": "rps_v2",
        "render_modes": ["ansi", "human"],
        "render_fps": 60
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        # The names of the players.
        self.possible_agents = ["player_1", "player_2"]

        # Translation from "player_<idx>" to integer idx.
        self.agent_index = dict(
            zip(self.possible_agents, [i + 1 for i in range(len(self.possible_agents))])
        )

        # Actions are [from, to, capture]; 0 means ignore, 1 - 24 are board positions.
        self._action_space = gym.spaces.MultiDiscrete(np.array([25, 25, 25]))

        # Observation is an array of board positions: 0 - empty, 1 - player_1, 2 - player_2.
        self._observation_space = gym.spaces.Box(
            low=0, high=2,
            shape=(25,),
            dtype=np.uint8
        )

        # The model gets created at reset.
        self._model = None

        # If render mode is human, initialize pygame.
        if render_mode == "human":
            global pygame
            import pygame
            import pygame.gfxdraw

            pygame.init()
            self._surface = pygame.display.set_mode((700, 700))
            pygame.display.set_caption("Mill")
            self._clock = pygame.time.Clock()
            self._pygame_initialized = True

            self._render_positions = [
                (0, 0), (0, 3), (0, 6), (1, 1), (1, 3), (1, 5),
                (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2),
                (3, 4), (3, 5), (3, 6), (4, 2), (4, 3), (4, 4),
                (5, 1), (5, 3), (5, 5), (6, 0), (6, 3), (6, 6)
            ]

            self._animation = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_space

    def observe(self, agent):
        # All agents observe the same board.
        return np.array(self._model._board)
    
    def close(self):
        pass

    def _get_opponent(self, agent):
        # Return the name of the opponent agent.
        agent_idx = self.agent_index[agent]  # 1 or 2
        opponent_idx = 3 - agent_idx  # 1 -> 2, 2 -> 1
        return self.agents[opponent_idx - 1]

    def reset(self, seed=None, options=None):
        # Create a new model.
        self._model = MillModel()

        # Reset the environment variables.
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.num_moves = 0

        # Compute the legal moves for both players.
        self.legal_moves = {
            agent: np.array(self._model.legal_moves(self.agent_index[agent]))
                for agent in self.agents
        }

        # Set the info for both players.
        self.infos = {
            agent: {
                'agent': agent,
                'move': 1,
                'phase': 'placing',
                'legal_moves': self.legal_moves[agent]
            } for agent in self.agents
        }

        # Set up the first player.
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        # Get the current player and its opponent.
        agent = self.agent_selection
        opponent = self._get_opponent(agent)

        # Calling the step method after the game has finished has no effect.
        if self._model.game_over():
            gym.logger.warn(
                "You are calling step method after the game has finished."
            )
            return

        # Actions must be a numpy array.
        if action is not None:
            action = np.array(action)

        # Check, if action is legal.
        legal_moves = self.legal_moves[agent]
        if action is not None and not np.any(np.all(legal_moves == action, axis=1)):
            gym.logger.warn(
                "You are trying to execute an illegal move. A random legal move is chosen instead."
            )
            action = None

        # If action is none or not legal, choose a random legal action instead.
        if action is None:
            action = legal_moves[np.random.choice(legal_moves.shape[0])]

        # Assert the shape of the action.
        assert action.shape == (3,)

        # The previous reward has just been observed. Start anew.
        self._cumulative_rewards[agent] = 0

        # Make the move.
        move_info = self._model.make_move(self.agent_index[agent], action.tolist())

        # Set the rewards for both players.
        if move_info['pieces_captured'] > 0:
            self.rewards[agent] = 1
            self.rewards[opponent] = -1
        else:
            self.rewards[agent] = 0
            self.rewards[opponent] = 0

        # Add the rewards to the cumulative reward.
        self._accumulate_rewards()
        
        # If the last player played, count a move.
        if self._agent_selector.is_last():
            self.num_moves += 1

            # Check if the game is too long.
            self.truncations = {
                agent: self.num_moves >= 100 for agent in self.agents
            }
        
        # Check if the game is over.
        if self._model.game_over():
            self.terminations = {
                agent: True for agent in self.agents
            }

        # Compute the legal moves for the opponent.
        self.legal_moves[opponent] = np.array(self._model.legal_moves(self.agent_index[opponent]))
        
        # Update the agent's info.
        self.infos[agent]['phase'] = move_info['player_phase']
        self.infos[opponent]['move'] = self.num_moves + 1
        self.infos[opponent]['phase'] = move_info['opponent_phase']
        self.infos[opponent]['legal_moves'] = self.legal_moves[opponent]

        # Set the animation and render.
        if self.render_mode == 'human':
            self._animation = {
                'src': action[0],
                'dst': action[1],
                'captured': action[2],
                'player': agent
            }

        self.render()

        # Set the next player.
        self.agent_selection = self._agent_selector.next()
    
    def render(self):
        if self.render_mode is None:
            return

        # Print the ASCII board to the terminal.
        elif self.render_mode == "ansi":
            print(self._model)

        # Render the board using pygame.
        elif self.render_mode == "human":
            # If we animate the piece, run the animation.
            if self._animation is not None:
                # Get player info.
                player_idx = self.agent_index[self._animation['player']]
                opponent_idx = self.agent_index[self._get_opponent(self._animation['player'])]

                # Back up the current board.
                model_backup = self._model.clone()

                # Remove the destination piece. It will be rendered separately.
                self._model._board[self._animation['dst']] = 0

                # If a piece has been captured, add it back to the board.
                if self._animation['captured'] > 0:
                    self._model._board[self._animation['captured']] = opponent_idx

                if self._animation['src'] > 0:                  
                    # Set up motion coordinates.
                    (row, col) = self._render_positions[self._animation['src'] - 1]
                    p0 = (50 + col * 100, 50 + row * 100)
                    (row, col) = self._render_positions[self._animation['dst'] - 1]
                    p1 = (50 + col * 100, 50 + row * 100)
                else:
                    # Set up motion coordinates.
                    p0 = (350, 800) if player_idx == 1 else (350, -100)
                    (row, col) = self._render_positions[self._animation['dst'] - 1]
                    p1 = (50 + col * 100, 50 + row * 100)
                
                # Animate the move from source to destination.
                self._animate_board(p0, p1, player_idx)
            
                # Restore the current board.
                self._model = model_backup

                # Animate capturing the piece.
                if self._animation['captured'] > 0:
                    # Set up motion coordinates.
                    (row, col) = self._render_positions[self._animation['captured'] - 1]
                    p0 = (50 + col * 100, 50 + row * 100)
                    p1 = (350, 800) if player_idx == 1 else (350, -100)

                    # Animate the captured piece flying out.
                    self._animate_board(p0, p1, opponent_idx)

            # Otherwise just paint the board.
            else:
                self._paint_board()
                pygame.display.flip()
    
    def _paint_piece(self, x, y, color1, color2):
        surface = self._surface
        pygame.gfxdraw.filled_circle(surface, x, y, 30, color1)
        pygame.gfxdraw.aacircle(surface, x, y, 30, color1)
        pygame.gfxdraw.filled_circle(surface, x, y, 20, color2)
        pygame.gfxdraw.aacircle(surface, x, y, 20, color2)

    def _paint_board(self):
            surface = self._surface

            # Background
            surface.fill("tan")
            
            # Squares
            pygame.draw.rect(surface, "black", pygame.Rect((45, 45), (610, 610)), 10)
            pygame.draw.rect(surface, "black", pygame.Rect((145, 145), (410, 410)), 10)
            pygame.draw.rect(surface, "black", pygame.Rect((245, 245), (210, 210)), 10)

            # Cross
            pygame.draw.line(surface, "black", pygame.math.Vector2((50, 350)), pygame.math.Vector2((250, 350)), 10)
            pygame.draw.line(surface, "black", pygame.math.Vector2((450, 350)), pygame.math.Vector2((650, 350)), 10)
            pygame.draw.line(surface, "black", pygame.math.Vector2((350, 50)), pygame.math.Vector2((350, 250)), 10)
            pygame.draw.line(surface, "black", pygame.math.Vector2((350, 450)), pygame.math.Vector2((350, 650)), 10)
            
            # Diagonals
            pygame.draw.line(surface, "black", pygame.math.Vector2((50, 50)), pygame.math.Vector2((250, 250)), 14)
            pygame.draw.line(surface, "black", pygame.math.Vector2((450, 450)), pygame.math.Vector2((650, 650)), 14)
            pygame.draw.line(surface, "black", pygame.math.Vector2((50, 650)), pygame.math.Vector2((250, 450)), 14)
            pygame.draw.line(surface, "black", pygame.math.Vector2((450, 250)), pygame.math.Vector2((650, 50)), 14)

            # Circles and pieces
            for (i, (row, col)) in enumerate(self._render_positions):
                position = i + 1

                # Player 1 piece
                if self._model._board[position] == 1:
                    self._paint_piece(50 + col * 100, 50 + row * 100, (128, 0, 64), (192, 0, 0))
                
                # Player 2 piece
                elif self._model._board[position] == 2:
                    self._paint_piece(50 + col * 100, 50 + row * 100, (128, 160, 0), (192, 192, 0))
                
                # Empty position
                else:
                    pygame.gfxdraw.filled_circle(surface, 50 + col * 100, 50 + row * 100, 10, (0, 0, 0))
                    pygame.gfxdraw.aacircle(surface, 50 + col * 100, 50 + row * 100, 10, (0, 0, 0))
        
    def _animate_board(self, p0, p1, player):
        global pygame

        # If episode is truncated, don't animate.
        if self.truncations[self.agent_selection]:
            return

        # Decode the starting and ending coordinates.
        (x0, y0) = p0
        (x1, y1) = p1

        # Choose the piece color based on the given player index.
        if player == 1:
            (color1, color2) = ((128, 0, 64), (192, 0, 0))
        else:
            (color1, color2) = ((128, 160, 0), (192, 192, 0))

        # Set the duration in frames.
        duration = self.metadata['render_fps']

        # Compute the step made in a single frame.
        dx = (x1 - x0) / duration
        dy = (y1 - y0) / duration

        # Run the animation.
        x = x0
        y = y0
        running = True

        while running:
            # Check pygame events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Truncate the episode.
                    self.truncations = {agent: True for agent in self.agents}
                    running = False
            
            # Draw the board and the animated piece.
            self._paint_board()
            self._paint_piece(round(x), round(y), color1, color2)
            pygame.display.flip()

            # Compute the next position of the animated piece.
            x += dx
            y += dy

            # Check the length of the animation.
            duration -= 1
            if duration <= 1:
                running = False

            # Wait next frame.
            self._clock.tick(self.metadata['render_fps'])