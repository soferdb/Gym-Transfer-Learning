# Gym-Transfer-Learning
Solving Cartpole-v1, Acrobot-v1, MountainCarContinuous-v0 using Policy-gradient methods, transfer learning, and simplified progressive-network.

Abbreviations: Transfer Learning (TL), Cartpole-v1(CP-v1), Acrobot-v1 (AB-v1), MountainCarContinuous-v0 (MCC-v0), Simplified Progressive Network (SPN)

The main difference between the 'simplified' version of the progressive network and the real progressive network is that in the simplified version, we use 2 normal pretrained networks instead of cascading the two, i.e., the second network is independent with the first network.
An illustration of our network architectures are shown in the following figures.

'Normal' ActorCritic Networks:

<img width="400" alt="Actor" src="https://user-images.githubusercontent.com/49614331/151966178-1755f587-0475-4e53-85b6-4ad53067da36.png">          <img width="400" alt="Critic" src="https://user-images.githubusercontent.com/49614331/151966191-91fa36d6-e2a4-484d-aa8a-5f6ee21b53bd.png">

These architectures were also used as the pretrained models.
The 'real' progressive network:

<img width="400" alt="ProgressiveNetwork" src="https://user-images.githubusercontent.com/49614331/151966431-c9e3591f-fb4a-483c-9085-3dc1673e45c1.png">

Our simplified version of progressive network:
High Level view:

<img width="300" src="https://user-images.githubusercontent.com/49614331/151966945-ba966693-a0b4-43a4-8f05-21483d5e354a.png"> 

Detailed view: (Orange layers are the normal Actor/Critic layers shown above)

<img width="600" src="https://user-images.githubusercontent.com/49614331/151967040-2c3f719b-6f06-46cf-9499-baacfa62cb5c.png">





Cartpole-v1:
Number of episodes and run-time until convergence 

Weight-Source| Episodes | Time
---| ---| ---
From scratch | 871 | 6h 21m
TL from AB-v1 | 452 | 3h 38m
TL from MCC-v0|663|5h 39m
SPN from AB-v1 and MCC-v0| 1333 | 22h 21m


Average score over 100 episodes trend graphs:

![CartpoleQ3_](https://user-images.githubusercontent.com/49614331/152200916-b755d5c1-8a34-41d3-bb89-1c4c35c9035e.png)



MountainCarContinuous-v0:

In this environment we used several methods to aid convergence, since this task is known for its problematic reward system. For implementation details see the attached document.

Number of episodes and run-time until convergence 

Weight-Source| Episodes | Time
---| ---| ---
From scratch | 5060 | 8h 47m
TL from CP-v1 | 2240 | 7h 11m
SPN from AB-v1 and CP-v1| 1341 | 6h 41m

Average score over 100 episodes trend graphs:

![MountainQ3_](https://user-images.githubusercontent.com/49614331/152201640-6889f62a-ae65-4509-a493-65adb596186c.png)




We invite the reader to read the attached document for more information and analysis.

