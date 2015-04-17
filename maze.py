import numpy as np

N = 0
E = 1
S = 2
W = 3
class Maze(object):

    def __init__(self, maze=None, start=None, goal=None, rewards=None):
        if maze is None:
            self.maze = np.asarray(
               [[0,1,0,0,0,0,0,0,0],
                [0,1,0,0,1,0,2,1,1],
                [0,0,0,0,1,0,1,0,1],
                [1,1,1,0,1,0,1,0,1],
                [1,0,0,0,1,0,1,0,1],
                [1,1,1,1,1,0,1,0,1],
                [1,0,0,0,0,0,0,0,0]]
            )
        if start is None:
            start = (0,0)

        if goal is None:
            goal = (6, 8)

        if rewards is None:
            self.rewards = [1, -0.01, -2]

        print self.get_moves((1,0))


    def transition_matrix(self):
        """
        Returns a transition matrix for each of the actions
        """
        num_states = self.maze.shape[0] * self.maze.shape[1]

        transition_matrix = np.zeros((4, num_states, num_states))

        # get all possible locations within the maze
        pos = [(i, j) for i in range(self.maze.shape[0]) for j in range(self.maze.shape[1])]

        # for each possible position get the transition probabilities
        for p in pos:
            next = self.get_moves(p)
            current_state = self.get_state(p)

            for a in range(4):
                next_states = []
                pass

    def get_state(self, pos):
        return self.maze.shape[1] * pos[0] + pos[1]

    def get_moves(self, position):
        """
        Get the possible moves from a given position
        """
        moves = []

        x = position[0]
        y = position[1]


        # check to see if north is a valid move
        if (x > 0) and (self.maze[x-1][y] != 1):
            moves.append((x-1, y))
        else:
            moves.append((x, y))

        # east
        if (y < self.maze.shape[1] - 1) and (self.maze[x][y+1] != 1):
            moves.append((x, y+1))
        else:
            moves.append((x, y))

        # south

        if (x < self.maze.shape[0] - 1) and (self.maze[x+1][y] != 1):
            moves.append((x+1, y))
        else:
            moves.append((x,y))

        # west
        if (y > 0) and (self.maze[x][y-1] != 1):
            moves.append((x, y-1))
        else:
            moves.append((x,y))

        return moves
