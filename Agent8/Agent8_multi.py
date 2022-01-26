#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# ## Agent 6

# Assumptions:
# 1. All cells are unblocked, until it observes them to be blocked.
# 2. All path planning is done with the current state of knowledge of the gridworld under this freespace assumption
# 3. There are three terrain types for unblocked cells: Flat(P), Hilly(H), Forest(F)
# 4. Each terrain type has false negative rate: 0.2, 0.5, 0.8 respectively. There are no false positives.
# 
# Data Structures:
# 1. grid (2D array): represents the full knowledge of the gridworld. Contains all blocked and terrain type of unblocked cells present.
# 2. agent grid (2D array): represents the current knowledge of the agent. Gets updated according to the field of view of the agent
# 3. pq (Priority Queue): will store the visited cells and will return them in increasing order of priority
# 4. dirx[], diry[] : will store values to calculate the neighbouring nodes in N,S,E,W direction.
# 5. p_containing (dictionary) : probability of the cell (key of dictionary) containing the target; gets updated when a block is encountered along the planned path or agent is not able to find target in the cell of 'assumed target'.
# 6. fn (dictionary) : false negative values of the cell (key of dictionary) which are updated while travelling along the planned path (while discovering the cells).
# 
# Pseudocode:
# 1. Initialize each cell with probability 1/(dim x dim).
# 2. Choose cells with highest probability first; if there are multiple cells with same probability then check for cell with lowest manhattan distance from the S. If again there are multiple cells with same distance then choose any cell as random. This cell will be 'assumed target' T.
# 3. Start with S.
# 4. Plan a path from S to T using the Manhattan distance heuristic.
# 5. Start traversing on the path until a) you reach the 'assumed target' or b) you get blocked in that path.
# 6. Once agent reaches the 'assumed target'; then agent will examine that cell for finding target.
# 7. If you find the target, terminate the program and find the path.
# 8. If you fail to find the target, update probabilites of every cell and repeat from step 2 with S as the most recent unblocked cell and 'assumed target' will change in step 2.
# 9. If you see a block, then update probabilities of every cell and repeat from step 2 with S as the most recent unblocked cell and target will remain same(due to scaling).

# Input:
# 1. n is the dimension of gridworld (nxn).
# 2. s is the string of length nxn that denotes the state of each cell in the grid world.
# 3. If s[i] is 'X', then the cell is blocked.
# 4. If s[i] is 'P', then the cell is unblocked with terrain type flat.
# 5. If s[i] is 'H', then the cell is unblocked with terrain type hilly.
# 6. If s[i] is 'F', then the cell is unblocked with terrain type thick forest.

# In[1]:


import numpy as np
from sortedcontainers import SortedSet
from math import sqrt, gcd
import random
import time
import matplotlib.pyplot as plt
# import seaborn as sns
import multiprocessing
import pickle
import csv

with open('P4data50x50srcs.pkl', 'rb') as f:
    sources = pickle.load(f)
with open('P4data50x50tgts.pkl', 'rb') as f:
    targets = pickle.load(f)

class MyPriorityQueue(object):
    def __init__(self, current_heuristic, target):
        self.current_heuristic = current_heuristic
        self._data = SortedSet()
        self.target = target
        self.g = dict()
        self.h = dict()
    def __len__(self):
        return len(self._data)
    def push(self, item):
        node = (self.calculate_priority(item), self.get_heuristic(item), item)
        self._data.add(node)
    def pop(self):
        node = self._data.pop(0)[2]
        return node
    def manhattan_distance(self, x, y): ### Manhattan Distance-> heuristic = |x0-y0| + |x1-y1|, x = [x0,x1] y = [y0,y1]
        return abs(x[0] - y[0]) +abs(x[1] - y[1])
    def euclidean_distance(self, x, y): ### Euclidean Distance-> heuristic = sqrt( (x0-y0)**2 + (x1-y1)**2 ), x = [x0,x1]
                                        ##  y = [y0,y1]
        return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    def chebyshev_distance(self, x, y): ### Chebyshev Distance-> heuristic = max(abs(x0-y0),abs(x1-y1)),
                                        ### x = [x0,x1] y = [y0,y1]
        return max(abs(x[0] - y[0]), abs(x[1] - y[1]))
    def get_heuristic(self, x):
        measure = self.current_heuristic
        if measure == 'm':
            self.h[x] = self.manhattan_distance(x, self.target)
        elif measure == 'e':
            self.h[x] = self.euclidean_distance(x, self.target)
        elif measure == 'c':
            self.h[x] = self.chebyshev_distance(x, self.target)
        return self.h[x]
    def calculate_priority(self, x):
        return self.g[x] + self.get_heuristic(x)

class GridWorld:
    def __init__(self, grid, start, target, n, m):
        '''
        Notations in grid: If grid contains '.', it means the cell is empty
                            and if the grid contains 'X', it means the cell is blocked
                            and if the grid contains 'P', it means the cell is of terrain type flat
                            and if the grid contains 'H', it means the cell is of terrain type hilly
                            and if the grid contains 'F', it means the cell is of terrain type forest
        '''
        self.n = n ## Number of columns
        self.m = m ## Number of rows
        self.start = start  ## Starting cell of the agent
        self.target = target ## Target cell of the agent
        self.trajectory = [] ## Trajectory followed by the agent
        self.grid = grid  ## Grid containing complete knowledge
        self.dirx8 = [0, 0, 1, -1, 1, -1, 1, -1] ## Directions used for sensing
        self.diry8 = [1, -1, 0, 0, 1, -1, -1, 1] ## Directions used for sensing
        self.dirx4 = [0, 0, 1, -1]  ## Directions used for traversing
        self.diry4 = [1, -1, 0, 0]  ## Directions used for traversing
            
        self.p_containing = {} #p_containing is the probability of (i,j) cell containing the target
        self.fn = {} #fn is the false negative of (i,j) cell 
        
class Agent8:
    def __init__(self, n, m, start, target, grid, agent_grid):
        self.n = n ## Number of columns in grid
        self.m = m ## Number of rows in the grid
        self.start = start ## The start cell of the grid
        self.target = target ## The target cell of the grid
        self.assumed_target = None ## target predicted by agent based on probability
        self.agent_grid = agent_grid  ## The current knowledge of the agent_grid
        self.dirx4 = [-1,1,0,0] ## 4 Directions for the agent to travel
        self.diry4 = [0,0,-1,1] ## 4 Directions for the agent to travel
        self.grid = grid # The full knowledge of the maze
        
        self.fn_p = 0.2 ## False negative for the cell with terrain type flat
        self.fn_h = 0.5 ## False negative for the cell with terrain type hilly
        self.fn_f = 0.8 ## False negative for the cell with terrain type forest
        self.examine_cost = 0
        self.movementGrid = []
        self.pfGrid = []
        self.fnGrid = []
        self.agent_x = []
        self.agent_y = []
        self.agent_nx = []
        self.agent_ny = []
        self.examine = []
        self.curr_source = ()
        self.movement_grid = np.full((n,m),0)
        self.pf_grid = np.full((n,m),0.0)
        
    def a_star(self, source, current_heuristic = 'm'):
#         self.movementGrid = np.full((self.m,self.n))
        
        '''
            A-star algorithm that plans path based on current knowledge of the agent
        '''
        dirx = [-1, 1, 0, 0] ## calculates the neighbours of the current node
        diry = [0, 0, 1, -1] ## calculates the neighbours of the current node
        visited = set() ## contains the nodes that have already been added in priority queue
        closed_list = set() ## contains the nodes that have been popped from priority queue
        open_list = MyPriorityQueue(current_heuristic, self.assumed_target) ## stores the visited nodes and pops it out 
                                                                    ## according to it's priority
        planned_path = [] ## Stores the path that the agent has planned according to it's current knowledge
        open_list.g[source]=0
        open_list.calculate_priority( source)
        open_list.push( source) ## This function runs in O(logn)
        visited.add(source) #visited[source] = 1
        parent = {} ## stores the parent of each node that is visited
        parent[source] = None
        while(len(open_list)>0):
            curr = open_list.pop()
#             self.sum_num_cells_processed+=1
            closed_list.add(curr)
            if(curr[0] == self.assumed_target[0] and curr[1] == self.assumed_target[1]):
                break
            for i in np.arange(4):
                childx = curr[0] + dirx[i]
                childy = curr[1] + diry[i]
                child = (childx, childy)
                if(childx>=0 and childx<self.m and childy>=0 and childy<self.n and (child not in closed_list) and self.agent_grid.grid[childx][childy]!='X'):
                    if(child not in visited):
                        visited.add(child)
                        parent[child] = curr
                        open_list.g[child] = open_list.g[curr]+1
                        open_list.calculate_priority(child)
                        open_list.push(child) 
                    else:
                        if open_list.g[curr]+1<open_list.g[child]:
                            parent[child] = curr
                            open_list._data.discard(child)
                            open_list.g[child] = open_list.g[curr]+1
                            open_list.calculate_priority(child)
                            open_list.push(child)
        if(self.assumed_target not in visited):
            return []
        curr = self.assumed_target
        while(curr != source):
            planned_path.append(curr)
            curr = parent[curr]
        planned_path.append(source)
        return planned_path[::-1]
    
    def makefnGrid(self, curr_cell):
        fn_grid = np.full((self.n,self.m),0)
        currx = self.curr_source[0]
        curry = self.curr_source[1]
        for i in range (0,self.m):
            for j in range (0,self.n):
                if((currx != i and curry != j) and self.grid.grid[currx][curry]!='X'):
                    fn_grid[i][j] = (1 - self.agent_grid.fn.get((i,j),0.5))*(self.m + self.n)/(abs(currx - i) +abs(curry - j))
                else:
                    fn_grid[i][j] = 0
        
        fn_grid[curr_cell[0]][curr_cell[1]] = 100 * fn_grid[curr_cell[0]][curr_cell[1]]
        self.fnGrid.append(fn_grid)
        
    def makepfGrid(self, curr_cell):
        self.pf_grid = np.full((self.n,self.m),0.0)
        currx = self.curr_source[0]
        curry = self.curr_source[1]
        for i in range (0,self.m):
            for j in range (0,self.n):
                if(currx != i and curry != j):
#                     print("#####################################################################################")
                    self.pf_grid[i][j] = 10000*(1 - self.agent_grid.fn.get((i,j),0.5))*(self.m + self.n)*self.agent_grid.p_containing[(i,j)]/(abs(currx - i) +abs(curry - j))
#                     print(self.pf_grid[i][j], " ")
#                     print(self.m," ",self.n)
#                     print("FN",self.agent_grid.fn.get((i,j),0.5))
#                     print("P",self.agent_grid.p_containing[(i,j)])
                else:
                    self.pf_grid[i][j] = 0
        if(curr_cell[0]==self.assumed_target[0] and curr_cell[1]==self.assumed_target[1]):
            self.examine.append(1)
        else:
            self.examine.append(0)
                    
        self.pf_grid[curr_cell[0]][curr_cell[1]] = 5 * self.pf_grid[curr_cell[0]][curr_cell[1]]
        self.pfGrid.append(self.pf_grid)
#         self.pf_grid = np.full((self.n,self.m),0)
        
#     def makemovementGrid(self):
#         movement_grid = np.full((self.n,self.m),0)
#         currx = self.curr_source[0]
#         curry = self.curr_source[1]
#         for i in range (0,self.m):
#             for j in range (0,self.n):
#                 if(currx != i and curry != j):
#                     pf_grid[i][j] = (1 - self.agent_grid.fn.get((i,j),0.5))*(m + n)*self.agent_grid.p_containing[(i,j)]/(abs(currx - i) +abs(curry - j))
#                 else:
#                     pf_grid[i][j] = 0
                    
#         self.pfGrid.append(pf_grid)
        
    def check_planned_path(self, planned_path):
        '''
            The agent will traverse along the planned path from current source to the 'assumed target' returned from the A*
            We will update false negative values according to terrain type for each (unblocked)cell traversed
            We update the value of restart node; if the agent reaches 'assumed target' the restart node will be current cell
            and if a blocked cell is encountered then te restart node will be the cell traversed before the blocked cell.
            If a blocked cell is encountered we will return the index of blocked cell also
        '''
        n =len(planned_path)
        traversed_path = []
        restart_cell = None
        block_cell = None
        block_encounter = False
#         self.movement_grid = np.full((n,m),0)
#         for i in range (0,self.m):
#             for j in range (0,self.n):
#                 if(self.agent_grid.grid[i][j]=='X'):
#                     self.movement_grid = -1

        for i in np.arange(n):
            cell = planned_path[i]
            currx = cell[0]
            curry = cell[1]
            
            self.agent_x.append(currx)
            self.agent_y.append(curry)
            self.makefnGrid(cell)
            self.makepfGrid(cell)
            
            self.agent_grid.grid[currx][curry] = self.grid.grid[currx][curry]
            
            if( self.grid.grid[currx][curry] == 'X' ):
#                 self.movement_grid[currx][curry] = -1
                block_cell = cell ## index of the cell blocked
                restart_cell = planned_path[i-1] ## restart cell will be the cell traversed before the blocked cell
                block_encounter = True ## flag when blocked cell is encountered along the planned path
                break
            else:
#                 self.movement_grid[currx][curry] = 1
                    
                if( self.grid.grid[currx][curry] == 'P' ):
                    self.agent_grid.fn[cell] = self.fn_p
                elif( self.grid.grid[currx][curry] == 'H' ):
                    self.agent_grid.fn[cell] = self.fn_h
                elif( self.grid.grid[currx][curry] == 'F' ):
                    self.agent_grid.fn[cell] = self.fn_f
                restart_cell = cell ## if the agent reaches the 'assumed target' then the restart cell will be current cell
            traversed_path.append(cell)
#             self.movementGrid.append(movement_grid)
            
        return traversed_path, restart_cell, block_encounter, block_cell
            
    def compute_path(self):
        '''
            First we will initialize the probability accross all cells of the grid and assume a target
            Initially all cells will have equal probability 1/dim*dim 
            So our target assumed will be one of the 4 neighbors (N,S,E,W) since all probabilities are equal
            and neighbors will have least distance; so any neighbor iis picked randomly.
            Priority for assuming the new target will be probability->distance->random choice
        '''
        target_found = False
        path = []
        curr = self.start
        self.initialize_probability() #initialize probability of all cells with 1/dim*dim
        self.assumed_target = self.new_target(self.start) #assume a target based on probability->distance->random
                                                        #initially the assumed target will be one of the 4 neighbors (N,S,E,W)
        while(True):
            planned_path = self.a_star(curr) #plan path using A* from curent cell to the 'assumed target'
#             print("Planned path: ", planned_path)
            if( len(planned_path) == 0 ):
#                 print("######: ", self.assumed_target)
                #if A* returns empty path it means that 'assumed target' is unreachable: 'it is as good as blocked' so update probability in same way when a block cell is encountered
                self.propogate_probability_foundblock(self.assumed_target)
                self.assumed_target = self.new_target(curr) #assume new target based on probability->distance->random
                continue
            
            #traverse along the planned path
            traversed_path, restart_node, block_encounter, block_cell = self.check_planned_path(planned_path)
            n = len(traversed_path)
            path.append(traversed_path)
            
            if(block_encounter == True):
                self.propogate_probability_foundblock(block_cell) # if block is encountered update probability accordingly
            elif(traversed_path[n-1] == self.assumed_target):
                self.examine_cost += 1
                if( self.assumed_target != self.target ):
                    self.propogate_probability_notfoundt() #if 'assumed target' is not actual target, update probability accordingly
                else:
                    if( random.uniform(0, 1) <= self.agent_grid.fn[(self.target[0],self.target[1])] ):
                        self.propogate_probability_notfoundt() #if agent fails to find target, update probability accordingly
                    else:
                        target_found = True
                
            if( target_found == False ):
                curr = restart_node #if target is not found, new source will be current cell or cell before blocked cell encountered on the planned path
                self.assumed_target = self.new_target(curr) #assume new target
            else:
                break
        return path
    
        
    def initialize_probability(self):
        '''
            We are taking free space assumption. So at time t=0 agent will hvae no knowledge of the grid.
            So we assume probability of all cells containing target to be distributed equally likely
            So we initialize probability of a cell containing target as 1/dim*dim
        '''
        for i in np.arange(self.m):
            for j in np.arange(self.n):
                self.agent_grid.p_containing[(i,j)] = 1/(self.m*self.n)
    
    def new_target(self, curr):
        '''
            When a block is encountered while traversing on the planned path or the agent fails to find the target.
            The agent will have to assume new target.
            The agent assumes new target on the basis of probabilistic knowledge gathered till that time.
            New target is chosen by giving following preference
                -> Cells with maximum probability of containing the target
                -> If there are more than one such cells; then choose cells with lowest manhattan distance from current source
                -> If again there are more than one such cells; then choose any cell at random
            This cell will be agent's new 'assumed target'
        '''
        self.curr_source = curr
        maxp = 0
        tmp = []
        tmp2 = []
        tar = None
        min_d = self.n + self.m - 2
        min_p = 1
#         print(self.probability_grid())
        utility = 0
        # tmp will have the cells with the highest value of utility function at that given time
        # Utility function is defined as probability of cell containing the target/ distance of that cell from the source
        # by this we consider both probability and distance as a factor for choosing our next target
        # If there is a cell with probability 0.2 at distance 20 and also a cell with probability 1.9 at distance 10;
        # Agent 6 and 7 will ignore the later cell; but with the utility function Agent 8 will first check the latter cell
        
        #Since probability is between 0 and 1; we normalize manhattan distance by dividing it with max distance possible.
        #Utility ->
        #If a cell is discovered and unblocked
            #-> (1-FN of cell i,j)*P(cell i,j containing the target)*200/(manhattan distance from source to cell)
        #else
            #-> P(cell i,j containing the target)*200/(manhattan distance from source to cell)
        for i in np.arange(self.m):
            for j in np.arange(self.n):
                if( curr == (i,j) ):
                    continue
                utility = (1-self.agent_grid.fn.get((i,j),0))*self.agent_grid.p_containing[(i,j)]*100/(abs(i - curr[0]) + abs(j - curr[1]))
                if( maxp < utility ):
                    maxp = utility
                    tmp = [(i,j)]
                elif( maxp ==  utility):
                    tmp.append((i,j))
#         print("Cells with highest proability: ", tmp)

        # from the cells in tmp we choose those cells with the least distance from the current source and store them in tmp2
        for i in range(len(tmp)):
#             p = self.agent_grid.p_containing[tmp[i]]
#             if( min_p > p ):
#                 min_p = p
#                 tmp2 = [tmp[i]]
#             elif( min_p == p ):
#                 tmp2.append(tmp[i])
            d = abs(tmp[i][0] - curr[0]) + abs(tmp[i][1] - curr[1])
            if( min_d > d ):
                min_d = d
                tmp2 = [tmp[i]]
            elif( min_d == d ):
                tmp2.append(tmp[i])
#         print("Cells with highest proability and smallest distance: ", tmp2)

        # from the cells in tmp2 we choose any one cell as random as our next 'assumed target'
        tar = random.choice(tmp2)
#         print("New target: ", tar)
        return tar
    
    def propogate_probability_notfoundt(self):
        '''
            The agent fails to find the target in the 'assumed target' cell in two cases:
                -> The target is actually not present in the 'assumed target' cell
                -> The target is present in the 'assumed target' cell but the agent fails to find it due to false negative result
            In either of these cases we will upate probability in the following way:
            Cell x,y is current cell where agent failed to find the target
            If cell x,y = cell i,j:
                P(Cell i,j contains the target | target not found at cell x,y) 
                    = FN of cell x,y * P(Cell x,y conatining target) / P0,0 + P0,1 + ... + FN*Px,y + ... + Pn,n
            If cell x,y != cell i,j:
                P(Cell i,j contains the target | target not found at cell x,y) 
                    = P(Cell x,y conatining target) / P0,0 + P0,1 + ... + FN*Px,y + ... + Pn,n
        '''
        denominator = 0
        numerator = 0
        cellx = self.assumed_target[0]
        celly = self.assumed_target[1]
        cell = (cellx, celly)

        # first we calculate the denominator for the probability to be updated
        for i in range(self.m):
            for j in range(self.n):
                if( self.assumed_target == (i,j) ):
                    denominator += self.agent_grid.fn[(i,j)] * self.agent_grid.p_containing[(i,j)]
                else:
                    denominator += self.agent_grid.p_containing[(i,j)]
                    
        # now we calculate numerator for the probability to be updated
        for i in range(self.m):
            for j in range(self.n):
                if( self.assumed_target == (i,j) ):
                    numerator = self.agent_grid.fn[(i,j)] * self.agent_grid.p_containing[(i,j)]
                    self.agent_grid.p_containing[(i,j)] = numerator/denominator #updating probability for all the cells
                else:
                    numerator = self.agent_grid.p_containing[(i,j)]
                    self.agent_grid.p_containing[(i,j)] = numerator/denominator #updating probability for all the cells
    
    def propogate_probability_foundblock(self, cell):
        '''
            If the agent finds a block while traversing the planned path, we update probabilities in the following way:
            Cell x,y is current cell where agent found the block
            If cell x,y = cell i,j:
                P(Cell i,j contains the target | block found at cell x,y) 
                    = 0
            If cell x,y != cell i,j:
                P(Cell i,j contains the target | block found at cell x,y) 
                    = P(Cell x,y conatining target) / P0,0 + P0,1 + ... + Pn,n (excluding Px,y)
        '''
        denominator = 0
        numerator = 0
        cellx = cell[0]
        celly = cell[1]
        cell = (cellx, celly) # cell which was found blocked in the path or the cell which is unreachable 
        
        for i in range(self.m):
            for j in range(self.n):
                if( self.assumed_target == (i,j) ):
                    continue
                else:
                    denominator += self.agent_grid.p_containing[(i,j)]
                    
        for i in range(self.m):
            for j in range(self.n):
                if( cell == (i,j) ):
                    numerator = 0
                    self.agent_grid.p_containing[(i,j)] = numerator/denominator
                else:
                    numerator = self.agent_grid.p_containing[(i,j)]
                    self.agent_grid.p_containing[(i,j)] = numerator/denominator
    
    def probability_grid(self):
        # display probability of containing the target for all the cells in the grid format 
        grid = []
        for i in np.arange(self.m):
            l=[]
            for j in np.arange(self.n):
                l.append(self.agent_grid.p_containing[(i,j)])
            grid.append(l)
        grid = np.array(grid)
        return grid


def calculate_path_length(ans_compute_path):
    ans = 0
    for x in ans_compute_path:
        ans = ans + len(x)-1
    return ans + 1
from tqdm import tqdm

def multi_process(grid):
# for s in grid_data:
    # cnt = 0
    # print(grid[1])
    s = grid[0]
    start = grid[1]
    target = grid[2]
    # return 1,2,3,4
    result_y = []
    # print("Started for p: ",0.3)
    total_time = 0
    examine_cost = 0
    movement_cost = 0
    total_cost = 0
    # print(1)
    # s = grid
    i = 0
    grid = [] ## matrix that contains full knowledge of gridworld
    m = 50
    n = 50
    # start = sources
    # target = targets
    # print("Started for grid with source ", start, "and target ", target)
    while(i<m*n):
        grid.append(list(s[i:i+n]))
        i = i + n
    grid = np.array(grid)

    start_time = time.time()
    gridObject = GridWorld(grid, start, target, n, m) ### Original grid knowledge object
    agentGridObject = GridWorld(np.full((m,n),'.'), start, target, n, m) ### Agent Grid Object
    grid_agent = Agent8(n, m, start, target, gridObject, agentGridObject)
    ans = grid_agent.compute_path()
    end_time = time.time()
    
    total_time = end_time - start_time
    movement_cost = calculate_path_length(ans)
    examine_cost = grid_agent.examine_cost
    total_cost = calculate_path_length(ans) + grid_agent.examine_cost
    
    return total_time, movement_cost, examine_cost, total_cost, grid_agent.agent_x, grid_agent.agent_y, grid_agent.fnGrid, grid_agent.pfGrid, grid_agent.movementGrid, grid_agent.examine


# In[ ]:


if __name__ == "__main__":
    cores = int(multiprocessing.cpu_count())
    # def multi_process():

    cnt = 0
    total_time = []
    examine_cost = []
    movement_cost = []
    total_cost = []
    avg_total_time = 0
    avg_examine_cost = 0
    avg_movement_cost = 0
    avg_total_cost = 0
    f = open('P4data50x50.txt', 'r')
    grid_data = f.readlines()
    grid_data = grid_data[0:50]
    data_pf = []
    data_fn = []
    data_ex = []
    data_x = []
    data_y = []
    data_nx = []
    data_ny = []
    data = []
    p = multiprocessing.Pool(5)
    for i in p.imap_unordered(multi_process,zip(grid_data,sources,targets)):
        cnt+=1
        print("Completed for grid:", cnt)
        print("Total Time: ", i[0], "Movement Cost: ", i[1], "Examine Cost: ", i[2], "Total Cost: ", i[3])
        total_time.append(i[0])
        movement_cost.append(i[1])
        examine_cost.append(i[2])
        total_cost.append(i[3])
        avg_total_time += i[0]
        avg_movement_cost += i[1]
        avg_examine_cost += i[2]
        avg_total_cost += i[3]
        agent_nx = i[4][1:]
    #     agent_x = i[4]
    #     agent_x = agent_x[:1]
    #     agent_nx.append(m-1)
        agent_ny = i[5][1:]
    #     tmp = i[7]
    #     print(i[7])
    #     i[7] = i[7][:-1]
    #     i[6] = i[6][:-1]
        ex = i[9][1:]
        # print(len(ex),len(agent_nx),len(agent_ny),len(i[7]),len(i[6]),len(i[4]),len(i[5]))
        for k in range(0,len(ex)):
            data_ex.append(ex[k])
            data_nx.append(agent_nx[k])
            data_ny.append(agent_ny[k])
            
        for j in range(0,len(i[7])-1):
            data_pf.append(i[7][j])
            data_fn.append(i[6][j])
            data_x.append(i[4][j])
            data_y.append(i[5][j])
        
        for b in range(0,len(ex)):
            data.append({'GridNumber': cnt, 'Agent_x': data_x[j], 'Agent_y': data_y[j], 'Pf': data_pf[j], 'Fn': data_fn[j], 'Agent_nx': data_nx[j], 'Agent_ny': data_ny[j], 'Examine': data_ex[j]})
            
    #         data.append({'GridNumber': cnt, 'Agent_x': i[7][j], 'Agent_y': i[8][j], 'AgentGrid': i[9][j], 'Agent_nx': agent_nx[j], 'Agent_ny': agent_ny[j]})
        if (cnt ==50):
            print(len(data_pf)," ",len(data_fn))
            break

    #     with open('Agent-8_avg_total_time.pkl', 'wb') as f:
    #         pickle.dump(avg_total_time, f)
    #     with open('Agent-8_avg_examine_cost.pkl', 'wb') as f:
    #         pickle.dump(avg_examine_cost, f)
    #     with open('Agent-8_avg_movement_cost.pkl', 'wb') as f:
    #         pickle.dump(avg_movement_cost, f)
    #     with open('Agent-8_avg_total_cost.pkl', 'wb') as f:
    #         pickle.dump(avg_total_cost, f)
    #     with open('Agent-8_total_time.pkl', 'wb') as f:
    #         pickle.dump(total_time, f)
    #     with open('Agent-8_examine_cost.pkl', 'wb') as f:
    #         pickle.dump(examine_cost, f)
    #     with open('Agent-8_movement_cost.pkl', 'wb') as f:
    #         pickle.dump(movement_cost, f)
    #     with open('Agent-8_total_cost.pkl', 'wb') as f:
    #         pickle.dump(total_cost, f)
    print("Average Total Time: ", avg_total_time/50)
    print("Average Examine Cost: ", avg_examine_cost/50)
    print("Average Movement Cost: ", avg_movement_cost/50)
    print("Average Total Cost: ", avg_total_cost/50)


    #data for assignment 4

    fieldnames = ['GridNumber', 'Agent_x', 'Agent_y', 'Pf', 'Fn', 'Agent_nx', 'Agent_ny', 'Examine']
    with open('data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    with open('DataAgent8_pf_1.pkl', 'wb') as f:
        pickle.dump(data_pf, f)
    with open('DataAgent8_fn_1.pkl', 'wb') as f:
        pickle.dump(data_fn, f)
    with open('DataAgent8_x_1.pkl', 'wb') as f:
        pickle.dump(data_x, f)
    with open('DataAgent8_y_1.pkl', 'wb') as f:
        pickle.dump(data_y, f)
    with open('DataAgent8_nx_1.pkl', 'wb') as f:
        pickle.dump(data_nx, f)
    with open('DataAgent8_ny_1.pkl', 'wb') as f:
        pickle.dump(data_ny, f)
    with open('DataAgent8_ex_1.pkl', 'wb') as f:
        pickle.dump(data_ex, f)


