class MillModel:
    # Define all possible mill triplets.
    mills = [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
        [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24],
        [1, 10, 22], [4, 11, 19], [7, 12, 16], [2, 5, 8],
        [17, 20, 23], [9, 13, 18], [6, 14, 21], [3, 15, 24],
        [1, 4, 7], [3, 6, 9], [16, 19, 22], [18, 21, 24]
    ]

    def __init__(self):
        self._board = [0 for _ in range(25)]
        
        # Player with index 0 is a dummy to avoid index shifting.
        self._player = [
            {'phase': 'dummy', 'pieces_holding': 0, 'pieces_playing': 0},
            {'phase': 'placing', 'pieces_holding': 9, 'pieces_playing': 0},
            {'phase': 'placing', 'pieces_holding': 9, 'pieces_playing': 0}
        ]

        # Compute the connections from the mills.
        self.connections = []
        for [a, b, c] in self.mills:
            self.connections.extend([[a, b], [b, c]])

    def clone(self):
        # Create and return a cuplicate of itself.
        board = MillModel()
        board._board = list(self._board)
        board._player = [dict(player) for player in self._player]
        return board

    def get_state(self):
        return self._board[1:]

    def get_phase(self, player):
        return self._player[player]['phase']

    def _in_mill(self, position):
        # Is the piece at the given position a part of a formed mill?
        for mill in self.mills:
            if position in mill:
                if self._board[mill[0]] == self._board[mill[1]] == self._board[mill[2]]:
                    return True
        return False

    def _all_pieces(self, player):
        # Return all the pieces of the given player.
        positions = []
        for (position, piece) in enumerate(self._board):
            if piece == player:
                positions.append(position)
        return positions

    def _free_pieces(self, player):
        # Return all pieces for of the given player that are not in a mill formation.
        positions = []
        for (position, piece) in enumerate(self._board):
            if piece == player and not self._in_mill(position):
                positions.append(position)
        return positions

    def _capture_pieces(self, player):
        # Return all pieces of the given player that can be captured.
        positions = self._free_pieces(player)
        if len(positions) == 0:
            positions = self._all_pieces(player)
        return positions

    def count_pieces(self, player):
        return len(self._all_pieces(player))

    def legal_moves(self, player):
        # If game has finished, there are no legal moves.
        if self.game_over():
            return []

        player_info = self._player[player]
        opponent = 2 if player == 1 else 1
        moves = []

        # If the player is in the placing phase.
        if player_info['phase'] == 'placing':
            # Check all board positions.
            for (dst, piece) in enumerate(self._board):
                # Can only place on any empty position.
                if dst == 0 or piece != 0:
                    continue

                # Do the move.
                self._board[dst] = player

                # If a mill has been formed, an opponent's piece must be captured.
                if self._in_mill(dst):
                    for piece in self._capture_pieces(opponent):
                        moves.append([0, dst, piece])
                
                # If a mill has not been formed, non-capturing move is possible.
                else:
                    moves.append([0, dst, 0])

                # Undo the move.
                self._board[dst] = 0

        # If the player is in the flying phase.
        elif player_info['phase'] == 'moving':
            # Check all board positions.
            for (src, src_piece) in enumerate(self._board):
                # Can only move it's own piece.
                if src == 0 or src_piece != player:
                    continue

                # Check all connections.¸
                for connection in self.connections:
                    if src not in connection:
                        continue

                    # Get the connecting position.
                    dst = connection[0] if src != connection[0] else connection[1]

                    # If destination is empty, the player can move the piece there.
                    if self._board[dst] == 0:
                        # Do the move.
                        self._board[src] = 0
                        self._board[dst] = player

                        # If a mill has been formed, an opponent's piece must be captured.
                        if self._in_mill(dst):
                            for piece in self._capture_pieces(opponent):
                                moves.append([src, dst, piece])
                        
                        # If a mill has not been formed, non-capturing move is possible.
                        else:
                            moves.append([src, dst, 0])

                        # Undo the move.
                        self._board[src] = player
                        self._board[dst] = 0

        # If the player is in the flying phase.
        elif player_info['phase'] == 'flying':
            # Check all board positions.
            for (src, src_piece) in enumerate(self._board):
                # Can only move it's own piece.
                if src == 0 or src_piece != player:
                    continue

                # Check all empty positions.
                for (dst, dst_piece) in enumerate(self._board):
                    # Can only move to an empty position.
                    if dst == 0 or dst_piece != 0:
                        continue

                    # Do the move.
                    self._board[src] = 0
                    self._board[dst] = player

                    # If a mill has been formed, an opponent's piece must be captured.
                    if self._in_mill(dst):
                        for piece in self._capture_pieces(opponent):
                            moves.append([src, dst, piece])
                    
                    # If a mill has not been formed, non-capturing move is possible.
                    else:
                        moves.append([src, dst, 0])

                    # Undo the move.
                    self._board[src] = player
                    self._board[dst] = 0

        return moves

    def make_move(self, player, move):
        (src, dst, take) = move

        player_info = self._player[player]
        opponent = 2 if player == 1 else 1
        captured = 0

        # If the player is in the placing phase.
        if player_info['phase'] == 'placing':
            self._board[dst] = player
            
            player_info['pieces_playing'] += 1
            player_info['pieces_holding'] -= 1
            
            if player_info['pieces_holding'] == 0:
                player_info['phase'] = 'moving'

        # If the player is in the moving phase.
        elif player_info['phase'] == 'moving':
            self._board[src] = 0
            self._board[dst] = player

        # If the player is in the flying phase.
        elif player_info['phase'] == 'flying':
            self._board[src] = 0
            self._board[dst] = player

        # If in any other phase, ignore the move.
        else:
            return {}
        
        # Get the information about the opponent.
        opponent_info = self._player[opponent]

        # If a piece is taken, consider what happens with the opponent.
        if take > 0:
            opponent_info['pieces_playing'] -= 1
            self._board[take] = 0
            captured = 1

            # The opponent goes from moving to flying.
            if opponent_info['phase'] == 'moving':
                if opponent_info['pieces_playing'] <= 3:
                    opponent_info['phase'] = 'flying'
            
            # The opponent goes from flying to losing.
            elif opponent_info['phase'] == 'flying':
                if opponent_info['pieces_playing'] <= 2:
                    opponent_info['phase'] = 'lost'
        
        # Check if the opponent can make moves.
        if len(self.legal_moves(opponent)) == 0:
            # If not, the opponent lost the game.
            opponent_info['phase'] = 'lost'
        
        # Return the info.
        move_info = {
            'player_phase': player_info['phase'],
            'opponent_phase': opponent_info['phase'],
            'pieces_holding': player_info['pieces_holding'],
            'pieces_playing': player_info['pieces_playing'],
            'pieces_captured': captured
        }

        return move_info
    
    def game_over(self):
        return self._player[1]['phase'] == 'lost' or self._player[2]['phase'] == 'lost'

    def __str__(self):
        b = self._board
        return (
            f"{b[1]}---------{b[2]}---------{b[3]}\n"
            f"| \\       |       / |\n"
            f"|   {b[4]}-----{b[5]}-----{b[6]}   |\n"
            f"|   | \\   |   / |   |\n"
            f"|   |   {b[7]}-{b[8]}-{b[9]}   |   |\n"
            f"{b[10]}---{b[11]}---{b[12]}   {b[13]}---{b[14]}---{b[15]}\n"
            f"|   |   {b[16]}-{b[17]}-{b[18]}   |   |\n"
            f"|   | /   |   \\ |   |\n"
            f"|   {b[19]}-----{b[20]}-----{b[21]}   |\n"
            f"| /       |       \\ |\n"
            f"{b[22]}---------{b[23]}---------{b[24]}"
        )