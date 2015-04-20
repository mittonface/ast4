import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt
from Tkinter import *
import time

    
class RLMaze(object):
    
    def __init__(self, maze=None, start=None, goal=None, R=None):
        if maze is None:
            self.maze = np.asarray([
                [0,1,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0],
                [0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1],
                [0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0],
                [0,0,2,2,2,2,0,0,0,0,2,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,1,0],
                [0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1],
                [1,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0],
                [0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0],
                [0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,2,2,2,2,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                [0,1,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,1,1,0,1,1,1,0,0,0,0,2,2,2,2,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                [0,1,1,1,1,1,0,0,0,0,0,0,0,2,2,2,2,0,0,0,1,0,0,2,2,2,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
                [0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,2,2,0,0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1],
                [0,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0],
                [0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1],
                [0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,0,0,1,1,1,0,1,1,0,0,2,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,2,0,2,2,2,2,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,1,0],
                [0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,2,2,2,2,1],
                [1,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0],
                [0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,0,2,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0],
                [0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,2,2,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                [0,1,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                [0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
                [0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1],
                [0,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,2,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1],
                [1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,2,2,2,2,2,2,0,0,2,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0],
                [0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,0,0,0,2,2,2,1,0,0,0,0,1,0,0,0,1],
                [0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                [0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1,1,1,0,1,1,1],
                [0,0,0,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
                [0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,0],
                [0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,2,2,2,2,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0],
                [0,1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0],
                [0,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,2,3]
                ])
        else:
            self.maze=maze


        self.x_bound = self.maze.shape[0] - 1
        self.y_bound = self.maze.shape[1] - 1
        self.n_states = self.maze.shape[0] * self.maze.shape[1]
        self.q_iters = 0

        if start is None:
            self.start = (0,0)
        else:
            self.start = start
            
        if goal is None:
            self.goal = (49, 49)
        else:
            self.goal = goal
            
        if R is None:
            self.reward_vals = [10, -0.05, -1]
        else:
            self.reward_vals = R

    def get_moves(self, pos):
        """
        Returns a list of the results of each action
        N, E, S, W
        """
        x = pos[0]
        y = pos[1]
        
        move_list = []
        
        # north
        if (x > 0) and (self.maze[x-1][y] != 1):
            move_list.append((x-1,y))
        else:
            move_list.append((x, y))
            
        # east
        if (y < self.y_bound) and (self.maze[x][y+1] != 1):
            move_list.append((x, y+1))
        else:
            move_list.append((x, y))
            
        # south
        if (x < self.x_bound) and (self.maze[x+1][y] != 1):
            move_list.append((x+1, y))
        else:
            move_list.append((x, y))
        
        # west
        if (y > 0) and (self.maze[x][y-1] != 1):
            move_list.append((x, y-1))
        else:
            move_list.append((x, y))
            
        return move_list
            
    def print_maze(self):
        print self.maze
        
    def get_transitions(self):
        
        n_states = self.maze.shape[0] * self.maze.shape[1]
        # now I need to build a matrix that has dimensions (A, S, S)
        # where
        # - the first dimension is the action that you can take.
        # - the second dimension is the starting state
        # - the probabilities of ending up in other states
        T = np.zeros((4, n_states, n_states))
        
        positions = [(x,y) for x in range(self.maze.shape[0]) for y in range(self.maze.shape[1])]

        
        for a in range(4):
            for p_start in positions:
                p_start_state = self.position_to_state(p_start)
                
                for p_end in positions:
                    moves = self.get_moves(p_start)                    
                    p_end_state =self.position_to_state(p_end)
                    if (self.position_to_state(moves[a]) == p_end_state):
                        T[a][p_start_state][p_end_state] = 1
                        
        T[:, self.position_to_state(self.goal), :] = 0
        T[:, self.position_to_state(self.goal), self.position_to_state(self.start)] = 1
            
            
  
        return T
            
    def get_rewards(self):
        n_states = self.maze.shape[0] * self.maze.shape[1]
        # [reward for getting to goal, step penalty]
        reward_vals = self.reward_vals
        
        # intialize with step penalties
        R = np.ones((n_states, 4)) * reward_vals[1]
        
        # reward for goal state
        R[self.position_to_state(self.goal),:] = reward_vals[0]
        
        # find the penalty spots in the maze
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                if self.maze[x][y] == 2:
                    R[self.position_to_state((x,y)),:] = reward_vals[2]

        return R
    
    def position_to_state(self, pos):
        # (0,0) is state 0, (0, 1) is state 1, (0, 2) is state 2
        # (1, 0) is state 3, (1, 1) is state 4
        x = pos[0]
        y = pos[1]
        
        return self.maze.shape[1] * x + y
        
    
    def draw_maze(self, policy=None, title=None):
        master = Tk()
        DRAWING_HEIGHT = 800
        DRAWING_WIDTH = 800
        num_horizontal = self.maze.shape[1]
        num_vertical = self.maze.shape[0]
        
        BOX_WIDTH = DRAWING_WIDTH // num_horizontal
        BOX_HEIGHT = DRAWING_HEIGHT // num_vertical
        
        w = Canvas(master, width=DRAWING_HEIGHT, height=DRAWING_WIDTH)
        if title is not None:
            master.wm_title(title)
        w.pack()

        # draw the empty grid
        for i in range(num_horizontal):
            for j in range(num_vertical):
                w.create_rectangle(i*BOX_WIDTH, j*BOX_HEIGHT, i*BOX_WIDTH+BOX_WIDTH, j*BOX_HEIGHT+BOX_HEIGHT, fill="white")

        # now want to loop through the maze and draw boxes the appropriate
        # colour
        x = 0
        y = 0
        for i in self.maze:
            for j in i:
                if j == 1:
                    w.create_rectangle(y*BOX_WIDTH, x*BOX_HEIGHT, y*BOX_WIDTH+BOX_WIDTH, x*BOX_HEIGHT+BOX_HEIGHT, fill="black")
                if j == 2:
                    w.create_rectangle(y*BOX_WIDTH, x*BOX_HEIGHT, y*BOX_WIDTH+BOX_WIDTH, x*BOX_HEIGHT+BOX_HEIGHT, fill="red")
                if j==3:
                    w.create_rectangle(y*BOX_WIDTH, x*BOX_HEIGHT, y*BOX_WIDTH+BOX_WIDTH, x*BOX_HEIGHT+BOX_HEIGHT, fill="green")
                if (y, x) == self.start:
                    w.create_rectangle(y*BOX_WIDTH, x*BOX_HEIGHT, y*BOX_WIDTH+BOX_WIDTH, x*BOX_HEIGHT+BOX_HEIGHT, fill="blue")

                y += 1
            x += 1
            y = 0
            

        if policy is not None:
            padding = 5
            x = 0
            y = 0
            for p in policy:
                if p == 0:
                    w.create_line(BOX_WIDTH/2 + (BOX_WIDTH*x), BOX_HEIGHT*y + padding, BOX_WIDTH/2 + (BOX_WIDTH*x), BOX_HEIGHT*y+BOX_HEIGHT - padding, arrow=FIRST, fill="orange")
                if p == 1:
                    w.create_line(x * BOX_WIDTH + padding, BOX_HEIGHT/2 + (BOX_HEIGHT*y), x*BOX_WIDTH+BOX_WIDTH-padding, BOX_HEIGHT/2 + (BOX_HEIGHT*y), arrow=LAST, fill="orange")
                if p == 2:
                    w.create_line(BOX_WIDTH/2 + (BOX_WIDTH*x), BOX_HEIGHT*y + padding, BOX_WIDTH/2 + (BOX_WIDTH*x), BOX_HEIGHT*y+BOX_HEIGHT - padding, arrow=LAST, fill="orange")
                if p == 3:
                    w.create_line(x * BOX_WIDTH + padding, BOX_HEIGHT/2 + (BOX_HEIGHT*y), x*BOX_WIDTH+BOX_WIDTH-padding, BOX_HEIGHT/2 + (BOX_HEIGHT*y), arrow=FIRST, fill="orange")

                x += 1
                if x == self.maze.shape[1]:
                    x = 0
                    y += 1
                
        mainloop()
        
    def qlearn(self, num_episodes=1000, gamma=.99, alpha_i=1, epsilon_i=1, epsilon_decay=0.999, alpha_decay=0.999):
        
        # initialize Q(s,a)
        Q = np.random.rand(self.n_states, 4)
        R = self.reward_vals


        for i in range(num_episodes):
            alpha = alpha_i
            epsilon = epsilon_i
            
            for start_state in [(x,y) for x in range(self.maze.shape[0]) for y in range(self.maze.shape[1])]:
                s = start_state
                
                while s != self.goal:
                    self.q_iters += 1
                    # make a copy of the array, we will compare this to the new Q 
                    # array after the iteration. If there is no significant change
                    # we'll break 
                    old_q = np.empty_like(Q)
                    old_q[:] = Q
                    
                    # There are some cases where there are no valid moves from the start
                    # position. This trips things up. So break
                    if self.stuck_state(s):
                        break
            
                    # choose an action from the current state
                    current_state = self.position_to_state(s)
                    action = self.epsilon_greedy(epsilon, Q[current_state])

                    current_Q_val = Q[current_state][action]

                    # now take the action
                    new_state = self.simple_move(s, action)
                    
                    # get the reward for this new state
                    if new_state == self.goal:
                        r = R[0]
                    elif self.maze[new_state[0]][new_state[1]]==2:
                        r = R[2]
                    else:
                        r = R[1]

                    new_state_num = self.position_to_state(new_state)
                    
                    # Reassign Q(s, a) with the correct value update function
                    # get the Q value of the best action on the next state
                    next_state_Q = np.max(Q[new_state_num])
                    Q[current_state][action] = current_Q_val + (alpha * ( r+(gamma*next_state_Q) - current_Q_val))
            
                    # now consider the new state
                    s = new_state
                    
                    # decay our epsilon and alpha, I'm not 100% sure this is the 
                    # proper place to do this. But it seems to converge on correct
                    # ansert with this here
                    epsilon *= epsilon_decay
                    alpha *= alpha_decay 

                    # if there was no significant changes in Q, break.
                    if np.allclose(old_q, Q, 0.0001, 0.0001):
                        break

        return Q
        
    def simple_move(self, pos, action):
        """
        Deterministic. Given a position and an action return the one
        position that will result
        """
        x = pos[0]
        y = pos[1]
        
        move_list = []
        
        # north
        if action==0 and (x > 0) and (self.maze[x-1][y] != 1):
            return (x-1,y)
        
        # east
        if action == 1 and (y < self.y_bound) and (self.maze[x][y+1] != 1):
            return (x, y+1)

            
        # south
        if action == 2 and (x < self.x_bound) and (self.maze[x+1][y] != 1):
            return (x+1, y)
        
        # west
        if action == 3 and (y > 0) and (self.maze[x][y-1] != 1):
            return (x, y-1)
        
        return (x, y)

    def stuck_state(self, pos):
        """
        we randomly choose a state to explore. This could be
        in a state that only has walls. We want to do a random restart 
        if we're in one of those states
        """

        n = self.simple_move(pos, 0)
        e = self.simple_move(pos, 1)
        s = self.simple_move(pos, 2)
        w = self.simple_move(pos, 3)

        return n == e == s == w == pos
    
    def epsilon_greedy(self, epsilon, actions):
        """
        Return the max, or not, w/e
        """
        rand = np.random.random()

        if rand <= epsilon:
            # return a random action
            return np.random.randint(0, 4)
        else:
            # return the max action
            return np.argmax(actions)
        
    def find_reward(self, policy):
        """
        Given the policy, find the total reward that we get by following that 
        policy from the start state
        """
        current_position = self.start
        current_reward = 0
        iter = 0
        while current_position != self.goal:
            iter+=1
            state_num = self.position_to_state(current_position)
            
            # look up in the policy what to do at this position
            action = policy[state_num]
            
            current_position = self.simple_move(current_position, action)
    
            if self.maze[current_position[0]][current_position[1]] == 0:
                current_reward += self.reward_vals[1]
            elif self.maze[current_position[0]][current_position[1]] == 2:
                current_reward += self.reward_vals[2]
            elif self.maze[current_position[0]][current_position[1]] == 3:
                current_reward += self.reward_vals[0]
            if current_reward > 99999 or iter == 99999:
                return 99999

        return current_reward

            
            
            
            
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import PolicyIteration
from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners import Q
from pybrain.rl.learners.valuebased import ActionValueTable

from Tkinter import *
def run_maze(maze, title=""):
    T = maze.get_transitions()
    R = maze.get_rewards()
    discount = 0.90

    value_iteration = ValueIteration(T, R, discount)
    value_iteration.run()
    print "VITER REWARD", maze.find_reward(value_iteration.policy)
    print "VITER TIME", value_iteration.time
    print "VITER ITERS", value_iteration.iter
    maze.draw_maze(value_iteration.policy, title=title+"v")

    policy_iteration = PolicyIteration(T,R, discount)
    policy_iteration.run()
    print "PITER REWARD", maze.find_reward(policy_iteration.policy)
    print "PITER TIME", policy_iteration.time
    print "PITER ITERS", policy_iteration.iter
    maze.draw_maze(policy_iteration.policy, title=title+'p')


    s = time.time()
    Q = maze.qlearn()
    n = time.time()
    q_policy = []
    for state in Q:
        q_policy.append(np.argmax(state))

    maze.draw_maze(q_policy, title=title+'q')
    print "Q LEARN", maze.find_reward(q_policy)
    print "Q LEARN TIME", (n-s)
    print "Q ITERS", maze.q_iters



rewards1 = [1000, -0.05, -3]
rewards2 = [1000, -5, -20]
rewards3 = [0, 1, -2]
maze1 = RLMaze(R=rewards1)
maze2 = RLMaze(R=rewards2)
maze3 = RLMaze(R=rewards3)
# maze = RLMaze(maze=themaze, goal=goal, start=start, R=rewards3)
# maze.draw_maze()


# run_maze(maze1, title='r1')
run_maze(maze2, title='r2')
# run_maze(maze3, title='r3')







