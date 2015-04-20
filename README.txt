To run the experiments on the large maze you can run the commands

python ast4_1.py # for the first reward schema
python ast4_2.py # for the second reward schema
python ast4_3.py # for the three reward schema

Alternative reward schemas can easily be provided near line 440 of these scripts.
The format is

[goal_reward, step_reward, penalty_reward]

To run the experiments on the smaller maze with a default reward structure of 
[100, -0.04, -5]

python ast4.py 


Requirements are listed in requirements.txt
