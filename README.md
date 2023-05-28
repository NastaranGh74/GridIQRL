# GridIQRL
GridIQRL: A Reinforcement Learning-Based Tool for Distribution Network Reconfiguration

This is an RL-based tool for distribution network reconfiguration in power systems. The current version uses deep Q-learning and dueling deep Q-learning methods. To allow faster visualization of the results and use by non-AI experts, an interface has been added to the code.
Distribution network reconfiguration is the process of changing power distribution line states while maintaining the radial structure of the network and supplying all loads to minimize network loss and optimize voltage profile. In general, the RL agent should make the best line switching decisions considering the following three constraints:
Markup : 1. The network structure should remain radial
Markup : 2. All network loads must be supplied
Markup : 3. The voltage accross the network should remain between 0.95 p.u. and 1.05 p.u.

To prevent agent from taking detrimental actions in the system, action space is chosen in a way to fulfill constraints 1 and 2. This code has been tested with 33-, 119-, and 136-bus test systems. For more information please refer to our recently published papers: [A Comparative Study of Reinforcement Learning Algorithms for Distribution Network Reconfiguration With Deep Q-Learning-Based Action Sampling](https://ieeexplore.ieee.org/abstract/document/10040655)

![image](https://github.com/NastaranGh74/GridIQRL/assets/85129387/00193bf0-94bf-4840-b3d9-aec6c80fb382)
