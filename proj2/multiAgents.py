# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print(newScaredTimes)
        "*** YOUR CODE HERE ***"
        successorGameState.data.score=0
        min_dis_ghost=float("inf")
        for i in newGhostStates:
            if(i.getPosition()==newPos):#if the next position is a ghost, return -1
                successorGameState.data.score=-1
                return successorGameState.getScore()
            min_dis_ghost=min(self.hmdistance(i.getPosition(),newPos),min_dis_ghost)
        successorGameState.data.score-=1/min_dis_ghost#else score-=1/minimum distance to ghost
        if(len(newFood.asList())+1==len(currentGameState.getFood().asList())):#if the next position is a dot, return 1
            successorGameState.data.score=1
            return successorGameState.getScore()
        min_dis_dot=float("inf")
        for i in newFood.asList():
            min_dis_dot=min(self.hmdistance(i,newPos),min_dis_dot)
        if(min_dis_dot!=float("inf")):
            successorGameState.data.score+=1/min_dis_dot#else score+=1/minimum distance to dot
        return successorGameState.getScore()

    def hmdistance(self,pos1:tuple,pos2:tuple):
        return abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1])
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'): # initialize depth-2 tree; 2 steps
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        res=self.maxscore(gameState,1)
        return res
    
    # maxscore only calculates for pacman
    def maxscore(self,gameState:GameState,depth_val):
        if(gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        maxval=float("-inf")
        bestact=Directions.STOP
        for act in gameState.getLegalActions(0):
            successor=gameState.generateSuccessor(0,act)
            if maxval < max(maxval,self.minscore(successor,depth_val,1)):
                maxval=max(maxval,self.minscore(successor,depth_val,1))
                bestact=act
        if(depth_val==1):
            return bestact
        return maxval

    # minscore only calculates for ghosts
    def minscore(self,gameState:GameState,depth_val,n):
        if(gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        minval=float("inf")
        # if not arriving at the deepest depth
        if(depth_val<self.depth):
            for act in gameState.getLegalActions(n):
                successor=gameState.generateSuccessor(n,act)
                # if the ghost's successor is still min node
                if n<gameState.getNumAgents()-1:
                    minval=min(minval,self.minscore(successor,depth_val,n+1))
                # if the ghost's successor is the max node; the depth of the tree+1
                else:
                    minval=min(minval,self.maxscore(successor,depth_val+1))
        # if arriving at the deepest depth of the tree(the depth has several layers: one for an agent)
        else:
            if n<gameState.getNumAgents()-1:
                for act in gameState.getLegalActions(n):
                    successor=gameState.generateSuccessor(n,act)
                    minval=min(minval,self.minscore(successor,depth_val,n+1))
            else:
                for act in gameState.getLegalActions(n):
                    successor=gameState.generateSuccessor(n,act)
                    minval=min(minval,self.evaluationFunction(successor))

        return minval

    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        res=self.maxscore(gameState,float("-inf"),float("inf"),1)
        return res
    
      # maxscore only calculates for pacman
    def maxscore(self,gameState:GameState,alpha,beta,depth_val):
        if(gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        maxval=float("-inf")
        bestact=Directions.STOP
        for act in gameState.getLegalActions(0):
            successor=gameState.generateSuccessor(0,act)
            eval = self.minscore(successor,alpha,beta,depth_val,1)
            if maxval < max(maxval,eval):
                maxval=max(maxval,eval)
                bestact=act
            alpha = max(alpha,eval)
            if (beta < alpha):
                break
        if(depth_val==1):
            return bestact
        return maxval

    # minscore only calculates for ghosts
    def minscore(self,gameState:GameState,alpha,beta,depth_val,n):
        if(gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        minval=float("inf")
        # if not arriving at the deepest depth
        if(depth_val<self.depth):
            for act in gameState.getLegalActions(n):
                successor=gameState.generateSuccessor(n,act)
                # if the ghost's successor is still min node
                if n<gameState.getNumAgents()-1:
                    eval = self.minscore(successor,alpha,beta,depth_val,n+1)
                    minval=min(minval,eval)
                    beta = min(beta,eval)
                    if (beta < alpha):
                        break
                # if the ghost's successor is the max node; the depth of the tree+1
                else:
                    eval = self.maxscore(successor,alpha,beta,depth_val+1)
                    minval=min(minval,eval)
                    beta = min(beta,eval)
                    if (beta < alpha):
                        break
        # if arriving at the deepest depth of the tree(the depth has several layers: one for an agent)
        else:
            if n<gameState.getNumAgents()-1:
                for act in gameState.getLegalActions(n):
                    successor=gameState.generateSuccessor(n,act)
                    eval = self.minscore(successor,alpha,beta,depth_val,n+1)
                    minval=min(minval,eval)
                    beta = min(beta,eval)
                    if (beta < alpha):
                        break
            else:
                for act in gameState.getLegalActions(n):
                    successor=gameState.generateSuccessor(n,act)
                    eval = self.evaluationFunction(successor)
                    minval=min(minval,eval)
                    beta = min(beta,eval)
                    if (beta < alpha):
                        break

        return minval

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        res=self.maxscore(gameState,1)
        return res
    
    def maxscore(self,gameState:GameState,depth_val):
        if(gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        maxval=float("-inf")
        bestact=Directions.STOP
        for act in gameState.getLegalActions(0):
            successor=gameState.generateSuccessor(0,act)
            if maxval!=max(maxval,self.avescore(successor,depth_val,1)):
                maxval=max(maxval,self.avescore(successor,depth_val,1))
                bestact=act
        if(depth_val==1):
            return bestact
        return maxval

    def avescore(self,gameState:GameState,depth_val,n):
        if(gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        aveval=0
        if(depth_val<self.depth):
            for act in gameState.getLegalActions(n):
                successor=gameState.generateSuccessor(n,act)
                if n<gameState.getNumAgents()-1:
                    aveval+=self.avescore(successor,depth_val,n+1)
                else:
                    aveval+=self.maxscore(successor,depth_val+1)
        else:
            if n<gameState.getNumAgents()-1:
                for act in gameState.getLegalActions(n):
                    successor=gameState.generateSuccessor(n,act)
                    aveval+=self.avescore(successor,depth_val,n+1)
            else:
                for act in gameState.getLegalActions(n):
                    successor=gameState.generateSuccessor(n,act)
                    aveval+=self.evaluationFunction(successor)
        return aveval


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    We define three weights: dot_weight, ghosts_weight, scare_weight
    score += dot_weight * (1/min_dist_dot) + ghost_weight *(1/min_dist_ghost) 
    According to whether there is scare_time left, we add scare_weight*max(ScaredTimes)*(1/min_dis_ghost) to the score
    """
    "*** YOUR CODE HERE ***"
    dot_weight = 10
    ghost_weight = -10
    scare_weight = 200
    pacman_pos = currentGameState.getPacmanPosition()
    food_pos_list = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    score = currentGameState.data.score
    min_dis_ghost=float("inf")
    min_dis_dot=float("inf")
    for g in GhostStates:
        min_dis_ghost = min(manhattanDistance(pacman_pos,g.getPosition()),min_dis_ghost)
    for d_pos in food_pos_list:
        min_dis_dot = min(manhattanDistance(pacman_pos,d_pos),min_dis_dot)
    min_dis_dot = max(min_dis_dot,1)
    # if now the ghost and pacman are in the same square
    if min_dis_ghost == 0:
        if max(ScaredTimes) != 0:
            score += (dot_weight*(1/min_dis_dot)+scare_weight*max(ScaredTimes)) 
        else:
            return -1e7
    # ghost and pacman not in the same square
    else:
        if max(ScaredTimes) != 0:
            score += (dot_weight*(1/min_dis_dot) + scare_weight*max(ScaredTimes)*(1/min_dis_ghost))
        else:
            score += (dot_weight*(1/min_dis_dot) + ghost_weight*(1/min_dis_ghost))
    
    return score

# Abbreviation
better = betterEvaluationFunction
