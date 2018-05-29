# Route Planning with Reinforcement Learning


### Environment 
```bash
N: row
M: column 
```
### State:
```bash
X: row
Y: column
DIRECTION = {W, E, N, S}
T
POWER
```
### Action
```bash
Action = {S, T, L, R}
```


Experiment Plan:
1. Design different test cases:
    1.1 A simple grid work with some obstacles
    1.2 With tuning cost gird world
    1.3 With distance limitation
    1.4 With a dynamics environment: time variance like traffic
2. Implement different method
    2.1 A* Dijkstra
    2.2 Tabular Reinforcement learning: Q learning
    2.3 Function approximation RL
        2.3.1 Linear or non-linear function of Q
        2.3.2 DNN
3. Baseline 
    3.1 on test case 1.1: use dijkstra is enough
    3.2 on test case 1.2: use A* with a heuristic
    3.3 on test case 1.3: use A* with a heuristic
    3.4 on test case 1.4: use A* with a heuristic
4. Our method
    4.1 on test case 1.1: use a tabular based RL
    4.2 on test case 1.2 and 1.3: use linear or non-linear DRL
    4.3 on test case 1.4 use NN based RL