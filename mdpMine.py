#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/vbipin/aip/blob/master/mdp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


#we plan to implement some of the algorithms related to MDPs and RL
#MDP study
#%matplotlib inline
#import matplotlib
#import numpy as np
#import matplotlib.pyplot as plt

#I am trying to avoid the numpy dependencies

import random
#
#We plan to implement the gridworld class 
#


# In[2]:


#Let us have a gridworld
#ref: Chapter 17, Artificial Intelligence a Modern Approach
#ref: CS188 https://inst.eecs.berkeley.edu/~cs188/fa19/
#ref: https://inst.eecs.berkeley.edu/~cs188/fa19/assets/slides/lec8.pdf
#ref: https://courses.cs.washington.edu/courses/cse473/13au/slides/17-mdp-rl.pdf

#This class will create a 2D grid of row x colums 
#Some of the cells can be disabled by putting it into walls
#cells are addressed just like 2d arrays (r,c)
#There are possibly many terminal states
#terminal states have only one action available: Exit 
#Transistion is as per the book 80% action and 20%sideways ( a variable noise is used to control this distribution)
#There is a special end state, (-1,-1), from which NO action is available. This state is used as a final state.

#Actions #just some alias
Up    = 0
Down  = 1
Right = 2
Left  = 3
Exit  = 4

class GridWorld :
    #Default is as given in the AIMA book
    def __init__(self, 
                 rows    =3, 
                 columns =4, 
                 walls   =[(1,1)], terminals= {(0,3):+1.0, (1,3):-1.0}, 
                 gamma   =1.0, 
                 living_reward=0,
                 noise   =0.2
                ) :
        """We dont expect these parameters to change during the agent run"""
        self.rows      = rows
        self.columns   = columns
        self.N         = rows * columns #total cells
        self.walls     = walls
        self.terminals = terminals #dictionary of terminal celss and their rewards.
        self.gamma     = gamma
        self.living_reward = living_reward
        self.all_actions   = [ Up, Down, Right, Left, Exit ]
        self.end_state     = (-1, -1) #a dummy state to reach after taking Exit
        self.all_states    = [(r,c) for r in range(rows) for c in range(columns) if (r,c) not in walls ] + [self.end_state]
        self.noise         = noise
        
        
        #transitions from each state and the probabilities
        self.noise                = noise
        self.action_transitions   = { 
            Up:   ([Up,    Left, Right], [1-noise, noise/2, noise/2 ]),
            Down: ([Down,  Left, Right], [1-noise, noise/2, noise/2 ]),
            Left: ([Left,  Up,   Down ], [1-noise, noise/2, noise/2 ]),
            Right:([Right, Up,   Down ], [1-noise, noise/2, noise/2 ]),
            Exit :([Exit], [1.0])
        }
    
    def actions(self, state) :
        """returns all valid actions from the current state"""
        if state in self.terminals :
            return [Exit]
        if state == self.end_state :
            return [] #No action available.
        return [ Up, Down, Right, Left ]
    
    def reward(self, state, action, next_state=None) :
        """reward is the instantaneous reward. It is usually R(s,a,s')"""
        #In grid world the reward depends only on state.
        if state in self.terminals :
            return self.terminals[state] #dict has the terminal values +1 or -1
        if state == self.end_state :
            return 0.0
        return self.living_reward        #usually a small -ve value
    
    def transitions(self, state, action) :
        """returna list of tuple(nextstate, action, probability)"""
        actual_actions, probs = self.action_transitions[action]
        return [ self._next_cell(state, a) for a in actual_actions ], actual_actions, probs
    
    def move(self, state, action) :
        """Take the action and return the tuple(new_state, reward, is_terminal)"""                          
        assert action in self.actions(state) #just a check if this is a valid action at this time or not
        
        cells, actions, p = self.transitions(state, action)
        
        #we choose one cell acccording to probabilities
        new_state   = random.choices(cells, weights=p)[0] #only one; we take index 0                
        reward      = self.reward(state, action) #
        
        is_terminal = False
        if new_state == self.end_state :
            is_terminal = True
            
        return new_state, reward, is_terminal #keep the same for mat as OpenAI gym.
    
    def _next_cell(self, state, action) : 
        """Blindly takes the action without checking anything and returns the position"""
        r,c = state #row & column
        if action == Exit :
            return self.end_state
        if action == Up :
            target = r-1, c  
        if action == Down :
            target = r+1, c
        if action == Right :
            target = r, c+1  
        if action == Left :
            target = r, c-1 
        
        if self._valid_cell(target) :
            return target
        return state #stay put the target is invalid.
    
    def _valid_cell(self, cell) :
        """Returns true if the cell is a valid cell"""
        r, c = cell #this may be an illegal node; we need to check
        
        #is it any of the walls?
        if (r,c) in self.walls :
            return False
        
        #is it outside the grid?
        if r < 0 or r >= self.rows or c < 0 or c >= self.columns :
            return False
        
        return True
    
    #pretty print the grid and agent if given.
    def print(self, agent_state=None) :
        for r in range(self.rows) :
            for c in range(self.columns) :
                cell = (r,c)
                if cell in self.walls :
                    print('# ', end='')
                elif cell in self.terminals :
                    if self.terminals[cell] > 0 :
                        print('+', end=' ')
                    else :
                        print('-', end=' ')
                elif cell == agent_state :
                    print('@ ', end='')
                else :
                    print('. ', end='')
            print("")


# In[ ]:





# In[3]:


grid_world = GridWorld(gamma=0.9, living_reward=-0.04)
start = (2,0) #as in the book


# In[4]:


# + and - are the terminal states. @ is our agent.
grid_world.print(start)


# In[5]:


#This is a simple class to hold the policy dictionary
#useful for printing the policy and hiding some details.

class Policy :
    def __init__(self, grid_world=None) :
        """Holds one policy and returns actions according to it"""
        self.grid_world = grid_world
        self.policy     = { } #{ state: policy_action}
        
    def __getitem__(self, state) :
        return self.policy[state]
    
    def __setitem__(self, state, action) :
        self.policy[state] = action
    
    
    
    #Just a pretty print function for easy debugging
    def print(self) :
        print_chars = {Up:'^', Down:'v', Right:'>', Left:'<', Exit:'+'}
        for state in [(r,c) for r in range(self.grid_world.rows) for c in range(self.grid_world.columns)]:
            
            if state in self.grid_world.terminals :
                if self.grid_world.terminals[state] >= 0 :
                    print('+', end=' ') #positive reward terminal
                else :
                    print('-', end=' ') #-ve reward terminal
                    
            elif state not in self.policy :
                print('#', end=' ') #walls
            else :
                print(print_chars[self.policy[state]], end=' ') #directions >, <, ^, v
                
            if (state[1]+1) % self.grid_world.columns == 0 :
                print("") #just a newline


# In[6]:


###################################################################################
# Now we implement some algorithms 
###################################################################################


# In[7]:


def qvalue(grid_world, state, action, V) :
    """returns the Q value of the state action pair"""
    #  SUM [  P(s' | s, a) * ( R(s,a,s') + V(s2) ) ] of all s' from (s,a)
    next_states, actions, p = grid_world.transitions(state, action) 
    gamma = grid_world.gamma
    
    values = [ p[i] * ( grid_world.reward(state, actions[i], s) + gamma*V[s] ) 
              for i,s in enumerate(next_states) ]
    #print(values)
    #print(sum(values))
    return sum( values )

def max_qvalue(grid_world, state, V) :
    """returns the maximum of q values and its action"""
    q = [ (qvalue(grid_world, state, action, V), action) for action in grid_world.actions(state) ]
    #print(q)
    return max(q) #returns (value, action)

def value_iteration(grid_world, N=1000) :
    states = grid_world.all_states
    #epsilon = 0.0001 * (1-gamma)/gamma
    epsilon = 1e-10
    
    #initialize to 0
    #U = { s: 0 for s in states }
    V = { s: 0 for s in states }
            
    while True :
        #we keep tracof the maximum value change
        #if the maximum value change is less than a small value, epsilon, we can stop our iterations
        max_delta = 0 
        
        for state in states :
            if state != grid_world.end_state :
                
                qmax, qaction = max_qvalue(grid_world, state, V)

                delta = abs(qmax - V[state])
                max_delta = max( [ max_delta, delta] )#we keep them max of these tow values

                #update the Values
                V[state] = qmax
        
        #print(max_delta)
        if max_delta < epsilon : #we are not improving much. Converged?
            break
            
    return V
 
def policy_from_value(grid_world, V) :
    p = Policy(grid_world)
    for state in grid_world.all_states :
        if state != grid_world.end_state : #we dont have a policy for this state because no actions are valid
            qmax, qaction = max_qvalue(grid_world, state, V)
            p[state] = qaction
    return p


#example run
# grid_world = GridWorld(gamma=0.9, living_reward=-0.04)
# V = value_iteration(grid_world)
# p = policy_from_value(grid_world, V)
# p.print()


# In[8]:


#ref: http://incompleteideas.net/book/first/ebook/node43.html

def policy_evaluation( grid_world, policy ) :
    """This will run value iteration until convergence and return the converged Values"""
    states = grid_world.all_states
    gamma  = grid_world.gamma
    #epsilon = 0.0001 * (1-gamma)/gamma
    epsilon = 1e-7
    
    V = { s: 0 for s in states }
            
    while True : #we exit when less than epsilon diff is made
        max_delta = 0
        
        for state in states :            
            if state != grid_world.end_state : #we dont have a policy for this state because no actions are valid
                
                action = policy[state] #we run this policy            
                q      = qvalue(grid_world, state, action, V)           
                #print(q)

                delta = abs(q - V[state])
                max_delta = max( [ max_delta, delta] )#we keep them max of these tow values

                V[state] = q
            
        if max_delta < epsilon : #we are not improving much. Converged?
            break
                
    return V


def policy_improvement(grid_world, policy) :
    """Returns the new improved policy"""
    
    while True :
        improving = False
        
        #find the values for this policy
        V = policy_evaluation( grid_world, policy )
        
        #find the policy according to the new values we got
        new_policy = policy_from_value(grid_world, V)
    
        for state in grid_world.all_states :
            if state != grid_world.end_state : #we dont have a policy for this state because no actions are valid

                if policy[state] != new_policy[state] : #Do we have an improvement?
                    improving = True
                
        if not improving:
            return policy
            break
            
        policy = new_policy
        
        #for debug
        #print('_______')
        #policy.print()


# In[9]:


###############################################################################
###### Some test code. ########################################################


# In[10]:


def random_policy(grid_world) :
    
    #we need to choose a random action every time the policy is accessed
    #here we overload the getitem 
    #when the user says policy[state] they get a random action
    class _RandomPolicy(Policy) :
        def __getitem__(self, state) :
            return random.choice(grid_world.actions(state))
    
    p = _RandomPolicy(grid_world) 
    return p

def fixed_policy(grid_world) :
    p = Policy(grid_world)
    p.policy = {state: Up for state in grid_world.all_states if state != grid_world.end_state }
    p.policy.update({state:Exit for state in grid_world.terminals})
    #print(p.policy)
    return p

def good_policy(grid_world) :
    p = Policy(grid_world)
    p.policy = {
        (0,0):Right, (0,1): Right, (0,2): Right, (0,3) : Exit,
        (1,0):Up,    (1,1): Right, (1,2): Up,    (1,3) : Exit,
        (2,0):Up,    (2,1): Left,  (2,2): Up,    (2,3) : Left,
               }
    p.policy.update({state:Exit for state in grid_world.terminals})
    #print(p.policy)
    return p


# In[11]:


def run(grid_world, state, policy=None) :
    """runs a full episode and return the total reward"""
    rewards = []
    gamma = grid_world.gamma
    
    time_step = 0
    while True :
        action = policy[state]
        #a.print()
        #print(action)
        state, r, exited = grid_world.move(state, action)
        rewards.append(r * (gamma**time_step) ) #the further we go down, the less we value the reward
        if exited :
            break    
        time_step += 1
    return rewards


def expected_utility(grid_world, state, policy, N=100) :
    """run the policy till completion several times and return the expected utility"""
    s = 0.0
    for _ in range(N) :
        #from the same start state we run till completion, N times
        s += sum( run(grid_world, state, policy) )
    return s/N


# In[12]:


#page  651; AIMA Book
#The utilities of the states in the 4 × 3 world, calculated with γ = 1 and
#R(s) = − 0.04 for nonterminal states



N = 1000

grid_world = GridWorld(gamma=1.0, living_reward=-0.04)
policy = good_policy(grid_world)
policy.print()

for state in grid_world.all_states :
    if state != grid_world.end_state :
        print( expected_utility(grid_world, state, policy, N) )


# In[13]:


##### Lets run some value iteration and check the policy from it


# In[14]:


grid_world = GridWorld(gamma=0.9, living_reward=-0.04)


# In[15]:


V = value_iteration(grid_world)


# In[16]:


p = policy_from_value(grid_world, V)


# In[17]:


p.print()


# In[18]:


####### try policy iteration


# In[19]:



grid_world = GridWorld(gamma=0.5, living_reward=-0.05)

#Before policy
p = fixed_policy(grid_world)

#new policy
newp = policy_improvement(grid_world, p)


p.print()
print('_____')
newp.print()


# In[20]:


print(V)


# In[21]:


allStates=grid_world.all_states
V[(0,1)]
grid_world.actions((1,3))


# In[22]:


#grid_world.transitions()
grid_world.print() # print the grid world


# In[23]:


grid_world.reward((0,3),grid_world.actions((0,3))[0]) #get reword for state (0,3)


# In[30]:


grid_world.reward((0,1),grid_world.actions((0,1))[0],(0,2))


# In[31]:


grid_world.reward((0,1),grid_world.actions((0,1))[0],(1,1))


# In[25]:


grid_world.actions((0,1))


# In[26]:


grid_world.transitions((0,1),grid_world.actions((0,1))[0])


# In[27]:


# returns prob distribution of action resulting from state (0,1)
grid_world.transitions((0,1),grid_world.actions((0,1))[0])[2]


# In[45]:


V_mine={}
for key in grid_world.all_states: #initilize value of all state to zero
    V_mine[key]=0
V_mine


# In[185]:


def computeValue (probability,reward,gamma, ValueOfState, nextStates):
    val=[probability[i]*(reward+gamma*ValueOfState[nextState]) for i, nextState in enumerate(nextStates)]
    return sum(val)
    


# In[193]:


def computeQ(gridWorld,state,action, valueOfStates):
    reward = grid_world.reward(state,action)
    nextStates, _, probability = grid_world.transitions(state,action) 
    val=[probability[i]*(reward+grid_world.gamma*valueOfStates[nextState]) for i, nextState in enumerate(nextStates)]
    return sum(val)


# In[239]:


for state in grid_world.all_states:
    valueOfActs=[]
    for _ in range(100): # usually we should use a dalta >0 for convergence
        actionsOfState = grid_world.actions(state) # all the possible actions of a state
        #valueMaxAct # do i need to declare it?
        Q_ofActs=[]
        for act in actionsOfState: #gonna pick max of all acts
            Q = computeQ(grid_world, state, act, V_mine)
            Q_ofActs.append(Q)
            #print (act, ':', Q_ofActs)
        if not Q_ofActs:
            print ('now empty', state)
            pass
#            V_mine[state]=-1
        if len (Q_ofActs):
            V_mine[state]=max(Q_ofActs)
        
print (V_mine)
print (V)

# In[238]:


Q


# In[230]:


V_mine


# In[117]:


V


# In[ ]:




