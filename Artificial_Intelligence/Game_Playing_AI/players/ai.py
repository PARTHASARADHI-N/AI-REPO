import time
import math
import random
import numpy as np
from helper import *

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.visits = 0  # Number of times the node was visited
        self.wins = 0  # Track wins
        self.parent = parent
        self.children = []
        self.action = action  # Action taken to reach this node

    def add_child(self, child_node):
        """Adds a child node to the current node."""
        self.children.append(child_node)

    def update(self, result):
        """Updates win/visit statistics after simulation and propagates up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.update(result)


    def ucb(self, total_visits, exploration_param=1):
        """Calculates the ucb score for the node."""
        if self.visits == 0:
            return float('inf')  # Infinity score to favor unvisited nodes
        win_ratio = self.wins / self.visits
        exploration = exploration_param * math.sqrt(2*math.log(total_visits) / self.visits)
        return win_ratio + exploration


class AIPlayer:
    
    def __init__(self, player_number: int, timer):
      
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer

    def get_move(self, state: np.array) -> Tuple[int, int]:
        root = Node(state)
       
        def defense(state):
            root = Node(state)
            valid = get_valid_actions(root.state)
            edges = get_all_edges(len(root.state))
            corners = get_all_corners(len(root.state))
      
            for move in valid:
                copied = root.state.copy()
                copied[move[0]][move[1]] = self.player_number
                if check_win(copied, move, self.player_number)[0]:
                    return move
                copied = root.state.copy()
                copied[move[0]][move[1]] = 3-self.player_number
                if check_win(copied, move, 3-self.player_number)[0]:
                    return move

            for move1 in valid:
                for move2 in list(set(valid)-set([move1])):
                    copied = root.state.copy()
                    copied[move1[0]][move1[1]] = self.player_number
                    copied[move2[0]][move2[1]] = self.player_number
                    if check_win(copied, move2, self.player_number)[0]:
                        return move2
                    copied = root.state.copy()
                    copied[move1[0]][move1[1]] = 3-self.player_number
                    copied[move2[0]][move2[1]] = 3-self.player_number
                    if check_win(copied, move2, 3-self.player_number)[0]:
                        if move1 in edges:
                            return move1
                        else:
                            return move2
            set_c =set(corners)-set(valid) 
            if set_c!= set(corners):
                dim = len(state)
                for m in range(len(state)):
                    for n in range(len(state)):
                        if state[m][n] == 3-self.player_number:
                            opp_neigh = get_neighbours(dim, (m, n))
                            for i in opp_neigh:
                                if i in corners and i in valid:
                                    return i
                for i in corners:
                      if i in valid:
                        return i
    
        def all_children_expanded(node):
            for child in node.children:
                if child.visits == 0:
                    return False
            return True
        
        def heuristics(move, node, player):
            dim = len(node.state)
            corners = get_all_corners(dim)
            edges = get_all_edges(dim)
            player_spot = []
            for i in range(len(node.state)):
                for j in range(len(node.state)):
                    if (node.state)[i][j] != 0 and (node.state)[i][j] != 3:
                        player_spot.append((i, j))
            
            our_neighborhood = []
            opponent_neighborhood = []
            for spot in player_spot:
                if (node.state[spot[0]][spot[1]]== player or node.state[spot[0]][spot[1]]== 3-player):
                   neigh = get_neighbours(dim, spot)
                   for neighbour in neigh:
                        if node.state[neighbour[0]][neighbour[1]] == player:
                             our_neighborhood.append(neighbour)
                if (node.state[spot[0]][spot[1]]== 3-player):
                   neigh = get_neighbours(dim, spot)
                   for neighbour in neigh:
                        if node.state[neighbour[0]][neighbour[1]] == 3-player:
                             opponent_neighborhood.append(neighbour)
            size = (len(node.state)+1)//2
            if (size == 4):
                 diagonal = [(1,1),(2,2),(3,3),(3,4),(3,5),(3,6),(0,3),(1,3),(2,3),(4,3),(5,3),(6,3),(3,0),(3,1),(3,2),(2,4),(1,5),(0,6)]
            if (size == 6):
                 diagonal = [(1,1),(2,2),(3,3),(4,4),(5,5),(5,6),(5,7),(5,8),(5,7),(5,8),(5,9),(5,0),(5,1),(5,2),(5,3),(5,4),(4,6),(3,7),(2,8),(1,9),(1,5),(2,5),(3,5),(4,5),(6,5),(7,5),(8,5),(9,5)]
            h_d =0
            h_n =0
            h_no=0
            h_edge = 0
            if (move in diagonal):
                h_d = 0.01
            if (move in our_neighborhood):
                h_n=0.2
            if (move in opponent_neighborhood):
                h_no=0.2
            if move in edges:
                h_edge = 0.2 
            return (h_edge+h_n+h_no+h_d)
        
        def select_node_to_expand(root):
            node = root
            heuristics_scaler = 1
            while node.children and all_children_expanded(node):
                children_ucb = {}
                for child in node.children:               
                    child_ucb = child.ucb(node.visits)
                    children_ucb[child] = child_ucb + heuristics_scaler*heuristics(child.action, child.parent,self.player_number)
                node = max(children_ucb, key=children_ucb.get)
            return node
    
        def expand_node(node, player_number):
            moves = get_valid_actions(node.state, player_number)
            visited_action = []
            for i in range(len(node.children)):
                visited_action.append(node.children[i].action)

            unvisited_moves = list(set(moves)-set(visited_action))
            if not moves:  # No valid moves available
                return None  # Return None if no expansion is possible

            if (len(unvisited_moves) > 0):
                for move in unvisited_moves:
                    child_state = node.state.copy()  # Copy current state
                    child_state[move[0]][move[1]] = player_number  # Apply move
                    child_node = Node(child_state, node, move)
                    node.add_child(child_node)
            l = {}
            for child in node.children:
                if (child.visits == 0):
                    l[child] = 0
            random_child_to_expand = random.choice(list(l.keys()))
            return random_child_to_expand
        
        def random_simulate(state, player_number):
            current_state = state.copy()
            current_player = player_number
            while True:
                moves = get_valid_actions(current_state, current_player)
                if not moves:  # No more moves possible
                    break
                random_move = random.choice(moves)
                current_state[random_move[0]][random_move[1]] = current_player
                if check_win(current_state, random_move, current_player)[0]:
                    return 1 if current_player == self.player_number else 0  # Win for AI or loss
                current_player = 3 - current_player  # Switch player
            return 0.5  # Draw or no win situation
       
        def backpropagate(node, result):
            node.update(result)
        
        def mcts(root, player_number, iterations=300):
            for _ in range(iterations):
                # Selection
                node_to_expand = select_node_to_expand(root)
         
                # Expansion
                child_node = expand_node(node_to_expand, player_number)
   
                if child_node is None:
                    continue  # Skip this iteration if no expansion was possible

                # Simulation
                result = random_simulate(child_node.state, player_number)

                # Backpropagation
                backpropagate(child_node, result)

            # Return the best action from the root node
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.action
        
        if defense(state):
            return defense(state)
        
        best_move = mcts(root, self.player_number)

        return best_move
