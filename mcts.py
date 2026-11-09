import math
import random


class MCTSNode:
    def __init__(self, model, player, parent=None, move=None):
        self.model = model                    # Cloned transition_model at this node
        self.player = player                  # Player whose turn it is to act (1 or 2)
        self.parent = parent                  # Parent MCTSNode
        self.move = move                      # Move taken from parent to reach here
        self.children = []                    # List of child MCTSNodes
        self.visits = 0                       # Visit count (N)
        self.wins = 0.0                       # Value / Wins (Q) for the player who played 'move'
        
        # Untried legal moves from this board state
        self.untried_moves = self.model.legal_moves(player=self.player)

    def is_fully_expanded(self):
        """Returns True if all legal moves from this state have been expanded."""
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """Returns True if the game is over or the current player is blocked."""
        return self.model.game_over() or len(self.untried_moves) == 0 and len(self.children) == 0

    def best_child(self, c_param=1.414):
        """
        Picks the child with the highest UCB1 score.
        Balances exploitation (win rate) with exploration.
        """
        best_score = -float("inf")
        best_nodes = []

        for child in self.children:
            # Exploitation: win rate of this child
            exploitation = child.wins / child.visits
            
            # Exploration: UCB1 bonus for less-visited nodes
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        # Break ties randomly
        return random.choice(best_nodes)


def determine_winner(model):
    """
    Evaluates who won in a terminal state or after rollout cutoff.
    Returns:
        1: Player 1 won
        2: Player 2 won
        0: Draw
    """
    p1_pieces = model.count_pieces(1)
    p2_pieces = model.count_pieces(2)

    # Check if a player lost by having fewer than 3 pieces after placement phase
    if p1_pieces < 3 and model.get_phase(1) != "placing":
        return 2
    if p2_pieces < 3 and model.get_phase(2) != "placing":
        return 1

    # Check if a player lost by being trapped (no legal moves)
    if len(model.legal_moves(1)) == 0:
        return 2
    if len(model.legal_moves(2)) == 0:
        return 1

    # If game ended on cutoff, whoever has more pieces wins
    if p1_pieces > p2_pieces:
        return 1
    elif p2_pieces > p1_pieces:
        return 2
    else:
        return 0


def rollout(model, start_player, max_depth=40):
    """
    Simulates a game to the end (or until max_depth) using random legal moves.
    """
    sim_model = model.clone()
    current_player = start_player
    depth = 0

    while not sim_model.game_over() and depth < max_depth:
        legal = sim_model.legal_moves(player=current_player)
        if not legal:
            # Current player is trapped and loses
            return 2 if current_player == 1 else 1

        # Make a random move
        move = random.choice(legal)
        sim_model.make_move(player=current_player, move=move)

        # Switch to opponent
        current_player = 2 if current_player == 1 else 1
        depth += 1

    return determine_winner(sim_model)


def mcts_search(root_model, player, iterations=300, max_simulation_depth=40, c_param=1.414):
    """
    Runs MCTS from the current board state and returns the best move.
    """
    legal_moves = root_model.legal_moves(player=player)
    
    # If there's 0 or only 1 legal move, don't waste time thinking
    if not legal_moves:
        return None
    if len(legal_moves) == 1:
        return legal_moves[0]

    root_node = MCTSNode(model=root_model.clone(), player=player)

    for _ in range(iterations):
        node = root_node

        # Selection: Traverse the tree to find a node to expand
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(c_param=c_param)

        # Expand the node if it's not terminal and has untried moves
        if not node.is_terminal() and len(node.untried_moves) > 0:
            move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
            
            # Create a new child node for this move
            child_model = node.model.clone()
            child_model.make_move(player=node.player, move=move)
            next_player = 2 if node.player == 1 else 1
            
            child_node = MCTSNode(
                model=child_model,
                player=next_player,
                parent=node,
                move=move
            )
            node.children.append(child_node)
            node = child_node

        # Simulate from the new node (or the terminal node if no expansion)
        winner = rollout(node.model, start_player=node.player, max_depth=max_simulation_depth)

        # Backpropagation: Update the win/visit counts up the tree
        curr = node
        while curr is not None:
            curr.visits += 1
            
            # 'curr.move' was executed by curr.parent.player
            if curr.parent is not None:
                move_maker = curr.parent.player
                if winner == move_maker:
                    curr.wins += 1.0
                elif winner == 0:
                    curr.wins += 0.5
                # loss adds 0.0

            curr = curr.parent

    # Pick the move that was visited the most from the root
    best_child = max(root_node.children, key=lambda c: c.visits)
    return best_child.move