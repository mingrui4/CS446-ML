# Assignment 12: Q-learning
Train an AI agent to play the Pong game using Q-learning. 

On the low level, the game works as follows: we receive the last 4 image frames which constitute the state of the game and we get to decide if we want to move the paddle to the left, to the right or not to move it (3 possible actions). After every single choice, the game simulator executes
the action and gives us a reward: either a +1 reward if the ball went past the opponent, a-1 reward if we missed the ball and 0 otherwise. Our goal is to move the paddle so that we
get lots of reward.
