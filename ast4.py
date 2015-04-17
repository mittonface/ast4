import sys, getopt
from maze import Maze

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "1")
    except:
        pass

    if '1' in args:
        maze_mdp()


def maze_mdp():
    from scipy import *
    import sys, time

    from pybrain.rl.environments.mazes import Maze, MDPMazeTask
    from pybrain.rl.learners.valuebased import ActionValueTable
    from pybrain.rl.agents import LearningAgent
    from pybrain.rl.learners import Q, SARSA
    from pybrain.rl.experiments import Experiment
    from pybrain.rl.environments import Task
    import matplotlib.pyplot as plt

    plt.gray()
    plt.ion()
    structure = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 0, 0, 1, 0, 0, 0, 0, 1],
                       [1, 0, 0, 1, 0, 0, 1, 0, 1],
                       [1, 0, 0, 1, 0, 0, 1, 0, 1],
                       [1, 0, 0, 1, 0, 1, 1, 0, 1],
                       [1, 0, 0, 0, 0, 0, 1, 0, 1],
                       [1, 1, 1, 1, 1, 1, 1, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # the first argument is the maze, the second is the goal tuple
    environment = Maze(structure, (7,7))
    # we have 81 different states and 4 actions
    controller = ActionValueTable(81,4)
    controller.initialize(1.)
    learner = Q()
    agent = LearningAgent(controller, learner)
    task = MDPMazeTask(environment)
    experiment = Experiment(task, agent)
    while True:
        experiment.doInteractions(100)
        agent.learn()
        agent.reset()

        plt.pcolor(controller.params.reshape(81,4).max(1).reshape(9,9))
        plt.draw()
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])