# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # outer loop: traverse all the iterations
        for iter in range(self.iterations):
            #print(self.values)
            new_value = util.Counter(self.values)
            # inner loop: traverse all the states and update their values
            for state in self.mdp.getStates():
                opt_action = self.computeActionFromValues(state)
                if opt_action!=None:
                    new_value[state] = self.computeQValueFromValues(state,opt_action)
                else:
                    new_value[state]=0
            self.values = new_value

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        # self.values[state]: int 
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        next_state_and_prob_list = self.mdp.getTransitionStatesAndProbs(state,action)
        Q_value = 0
        for next_state, prob in next_state_and_prob_list:
            Q_value += prob*(self.mdp.getReward(state,action,next_state)+self.discount*self.values[next_state])
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        max_Q = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            Q_value = self.getQValue(state,action)
            if Q_value > max_Q:
                max_Q = Q_value
                opt_action = action
        return opt_action # actual argmax may be a key not in counter()??????

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # compute predecessors of all states
        # key: state value: predecessor set of this state
        predecessor_dict = util.Counter() 
        for state in self.mdp.getStates():
            predecessor_dict[state] = set()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next_state,prob in self.mdp.getTransitionStatesAndProbs(state,action):
                    predecessor_dict[next_state].add(state)
        # initialize a priority queue
        priotity_queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state)==False:
                opt_action = self.getAction(state)
                max_Q_value = self.getQValue(state,opt_action)
                diff = abs(self.values[state]-max_Q_value)
                priotity_queue.update(state,-diff)
        # iteration
        for iter in range(self.iterations):
            if priotity_queue.isEmpty()==True:
                break
            state = priotity_queue.pop()
            if self.mdp.isTerminal(state)==False:
                opt_action = self.getAction(state)
                max_Q_value = self.getQValue(state,opt_action)
                self.values[state] = max_Q_value
            for pred in predecessor_dict[state]:
                opt_action_pred = self.getAction(pred)
                max_Q_value_pred = self.getQValue(pred,opt_action_pred)
                diff = abs(self.values[pred]-max_Q_value_pred)
                if diff > self.theta:
                    priotity_queue.update(pred,-diff)