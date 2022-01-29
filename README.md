# Voyage-Into-the-unknown
Built Artificial Intelligence agents based on different variant of the robot path planning algorithm called Repeated A-star algorithm.

The algorithm for the repeated A-star is as follows:
1. Initialise the agent's initial position and the target location. Also, according to the agent's knowledge, the current grid contains all unblocked cells.
2. The agent does the following steps until it reaches the target:
   - The robot runs the A-star algorithm, on the current knowledge of the grid, which will find the shortest path that the agent will have to travel in the current knowledge of the grid to reach the target.
   - The robot starts travelling on the path and update it's knowldge as it travels on the new cells.
   - If the current cell of the path is unblocked, then the robot goes into the cell and updates it's knowledge
   - If the current cell of the path is blocked, then the robot will stop in the previous cell and update the knowledge. After updating it's knowledge, the robot will replan the from the current cell. 

#### In this project, we designed the variations of A-star algorithms for different situations. We explored 9 different situations which leads to 9 different algoritms.

## Agent 1
The initial location of the agent is top left corner of the grid and the target location is the botton right of the grid. The agent will implement the above described repeated A-star algorithm to find the shortest path from the start position to the target. In this project, the agent can sense cells into 8 different directions (N, S, E, W, NE, NW, SE, SW) and update it's knowledge.

## Agent 2

Similar to Agent 1, the inital location of the agent is the top left corne of the grid and the target location is the bottom right of the grid. The agent will be in the same situation as Agent 1 but this agent can look sense the cell that it is currently present on. So, here the agent will enter the next cell of the path and if it is unblocked it will update it's knowledge and continue and if the cell is blocked, then the agent will move back to the previous cell and replan the path using A-star algorithm.

For details and analysis of Agent 1 and Agent 2 algorithm refer to this [report]().

## Agent 3

This agent uses a more advance algorithm than the one used by Agent 2. This agent detects the state of the cell it is present in and in addition to this, it also knows the number of cells blocked in the 8 cells in the neighbourhood of the current cell. The notation used to denote certain quantities are as follows:
1. Nx : the number of neighbors cell x has.
2. Cx : the number of neighbors of x that are sensed to be blocked.
3. Bx : the number of neighbors of x that have been confirmed to be blocked.
4. Ex : the number of neighbors of x that have been confirmed to be empty.
5. Hx : the number of neighbors of x that are still hidden or unconfirmed either way.

This agent uses the following rules for inferences:
1. If Cx = Bx, then all remaining hidden neighbours are empty
2. If Nx - Cx = Ex, then all remaining hidden neighbours of x are blocked
3. If Hx = 0, nothing nothing remains to be inferred for cell x

When the agent enters any non visited cell, it examines the current knowledge and checks if it can draw any conclusions using the above rules of inference.

## Agent 4

Following the notations described in the above agent, this agent uses more inference rules for drawing conclusions from it's current knowledge. The Agent uses the **method of substitution** and the **method of subsequence** to draw conclusion. To know more about these methods, refer to the report.

Details and Analysis of Agent 3 and Agent 4 is available [here]().

## Agent 6, 7

This agent uses the probabilistic methods to draw inference from it's current knowledge. The target cell is hiddent in this case. Hence, after entering each cell, the agent will sense whether the target is present in the cell or not. Also, the examination might yield False Negative results. But we know the probability that the examination will yield a False Negative result.  There are three types of cells in the terrain:
1. Forest: The probability of successfully finding the target in the forest cell is 0.2
2. Hilly :  The probability of successfully finding the target in the forest cell is 0.5
3. Flat: The probability of successfully finding the target in the forest cell is 0.8

The algorithm stops when the agent successfully finds the target.

Agent 5 uses the **probability of the cell containing the target** and the Agent 6 uses **probabilty of the agent successfully finding the target** to draw conslusions.


## Agent 8

For the above algorithm the agent prioritises going to the cell with maximum corresponding probability, but in Agent 8 the agent will prefer going to the cell which is near it's current location and has good probability. So, instead of prioritising only on the basis of probabiltiy, this agent prioritises on the ratio of probability and Manhattan distance.

For more detailed analysis and results of Agent 6, 7 and 8,  refer to the [report]().

## Agent 9

This agent uses Machine Learning models to mimic the actions done by Agent 1 and 2. The challenges for modelling this agent include, the data representation for learning and the hyperparameters for training the model like number of layers, number of neurons in each layer. This agent was modeled using Artificial Neural Network.

## Agent 10

This agent uses Machine Learning models to mimic the actions done by Agent 1 and 2. The challenges for modelling this agent include, the data representation for learning and the hyperparameters for training the model like number of layers, number of neurons in each layer. This agent was modeled using Convolutional Neural Network.
