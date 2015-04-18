import numpy as np
from scipy.sparse import csc_matrix
from scipy import *
class Pacman(object):
    def __init__(self):
        self.board = np.asarray([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
        [1,0,1,1,1,0,0,1,0,0,1,1,1,0,1],
        [1,0,1,1,1,0,0,1,0,0,1,1,1,0,1],
        [1,0,0,0,0,0,1,1,1,0,0,0,0,0,1],
        [1,1,0,1,1,0,1,1,1,0,1,1,0,1,1],
        [1,1,0,1,1,0,0,0,0,0,1,1,0,1,1],
        [1,0,0,1,1,0,1,1,1,0,1,1,0,0,1],
        [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,0,1,0,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,0,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,0,1,0,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ])
        
        # all unvisited spaces have a dot which is worth some reward
        self.dots = np.ones((self.board.shape[0], self.board.shape[1]))
        self.collected_dots = 0
        self.max_dots = self.board.shape[0] * self.board.shape[1]
        
        self.x_bound = self.board.shape[0] - 1
        self.y_bound = self.board.shape[1] - 1
        self.start = (1,1)
        
        # pacman wins if he collects 264 dots
        self.goal = 264

    def take_action(self, pos, a):
        """
        Alternative to get moves?
        Returns the single position that you end up in if you take an action
        from the start pos
        """

        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        move_list = []
        
        # north
        if (a==0) and (x > 0) and (self.board[x-1][y] != 1):
            return self.get_dot_at_position((x-1,y))
            
        # east
        if (a==1) and (y < self.y_bound) and (self.board[x][y+1] != 1):
            return self.get_dot_at_position((x, y+1))
            
        # south
        if (a==2) and (x < self.x_bound) and (self.board[x+1][y] != 1):
            return self.get_dot_at_position((x+1, y))
        
        # west
        if (a==3) and (y > 0) and (self.board[x][y-1] != 1):
            return self.get_dot_at_position((x, y-1, z))

        return pos
            
        
    def get_moves(self, pos):
        """
        Returns a list of the results of each action
        N, E, S, W
        """
        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        move_list = []
        
        # north
        if (x > 0) and (self.board[x-1][y] != 1):
            move_list.append(self.get_dot_at_position((x-1,y)))
        else:
            move_list.append(pos)
            
        # east
        if (y < self.y_bound) and (self.board[x][y+1] != 1):
            move_list.append((self.get_dot_at_position(x, y+1)))
        else:
            move_list.append(pos)
            
        # south
        if (x < self.x_bound) and (self.board[x+1][y] != 1):
            move_list.append(self.get_dot_at_position((x+1, y)))
        else:
            move_list.append(pos)
        
        # west
        if (y > 0) and (self.board[x][y-1] != 1):
            move_list.append(get_dot_at_position((x, y-1, z)))
        else:
            move_list.append(pos)
            
        return move_list
    
    def get_transitions(self):
        n_states = (self.board.shape[0] * self.board.shape[1]) ** 2
        
        # now I need to build a matrix that has dimensions (A, S, S)
        # where
        # - the first dimension is the action that you can take.
        # - the second dimension is the starting state
        # - the probabilities of ending up in other states
        T = np.empty((4, n_states, n_states)) 
        positions = [(x, y, z) for x in range(self.board.shape[0]) for y in range(self.board.shape[1]) for z in range(self.max_dots)] 

        
        for a in range(4):
            for p_start in positions:
                p_start_state = self.position_to_state(p_start)
                T[a][p_start_state][self.position_to_state(self.take_action(p_start, a))] = 1
  
        return T
        
    def get_rewards(self):
        n_states = self.board.shape[0] * self.board.shape[1]
        
        # [reward for all dots, reward per dot, ghost penalty]
        reward_vals = [100, 1, -50]
        
        # intialize with step penalties
        R = np.ones((n_states, 4)) * reward_vals[1]
        
        # reward for goal states
        all_goal_states = [(x,y,z) for x in range(self.board.shape[0])]
        R[self.position_to_state(self.goal),:] = reward_vals[0]
        
        # find the penalty spots in the maze
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.board[x][y] == 2:
                    R[self.position_to_state((x,y)),:] = reward_vals[2]

        return R        
    
    def get_dot_at_position(self, pos):
        """
        When we enter a position, we will either collect a dot or not
        if the position has already been visited there is no dot there
        and we dont increase the number of dots that we've gotten
        """
        if (self.dots[pos[0]][pos[1]] == 1):
            self.dots[pos[0]][pos[1]] = 0
            self.collected_dots += 1

        return (pos[0], pos[1], self.collected_dots)
                             
    def position_to_state(self, pos):
        """
        I need to map each possible position to an integer number
        """
                             
        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        return self.board.shape[1] * self.max_dots * x + \
               self.max_dots * y + \
               z
               
pm = Pacman()

T = pm.get_transitions()

