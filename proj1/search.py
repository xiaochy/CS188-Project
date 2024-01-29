# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):

        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# (successor, action, stepCost) state definition
# successor: a successor to the current state
# action: the action the current state need to get to its successor
# stepCost: incremental cost used to expand to that successor

def depthFirstSearch(problem: SearchProblem):
    # store all the nodes on the fringe (stack = fringe)
    stack=util.Stack()
    # the nodes which have been explored
    explored=set()
    # current state's successor; Initially, the successor is the start state
    ptr=(problem.getStartState(),'Stop',1) 
    # store the path we find
    res=[]
    # if successor != goal state, continue searching
    while not problem.isGoalState(ptr[0]):
        # if successor has been explored, we will not explore this node
        if(ptr[0] in explored):
            # pop successor from the stack
            stack.pop()
            # pop the successor from the path we constructed
            res.pop()
            # two steps: get the next node on the stack(second deepest node)
            ptr=stack.pop()
            stack.push(ptr)
            continue
        # if successor has not been explored->add successor to explored set(i.e.remove from the fringe)
        explored.add(ptr[0])
        # add successor to path
        res.append(ptr[1])

        # determine whether the successors of the current node have been explored
        for i in problem.getSuccessors(ptr[0]):
            # if not, push them onto the stack & let the last-pushed node be the next explored node
            if(i[0] not in explored):
                stack.push(i)
                ptr=i 
    # if successor is the goal state, add it to the path->return path->end 
    res.append(ptr[1])
    return res[1:]
        
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    print("Start:", problem.getStartState())
    print(problem.getSuccessors(problem.getStartState())[0][1])
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getSuccessors(problem.getStartState())[0][0]))

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    import copy
    fringe = util.Queue()
    explored = []
    ptr = (problem.getStartState(),[])
    fringe.push(ptr)
    # successor_path includes the current action
    explored.append(ptr[0])
    while not fringe.isEmpty():
        ptr = fringe.pop()
        if problem.isGoalState(ptr[0]):
            return ptr[1]
        for i in problem.getSuccessors(ptr[0]):
            if i[0] not in explored:
                explored.append(i[0])
                successor_path = copy.deepcopy(ptr[1])
                successor_path.append(i[1])
                fringe.push((i[0],successor_path))
    
    return ptr[1]

    #use queue to store all the node on the fringe
    # queue=util.Queue()
    # # the explored node; removed from the fringe
    # explored=set()
    # # store successor
    # ptr=(problem.getStartState(),'Stop',1)
    # # 
    # res_dict={}
    # # add start node as explored
    # explored.add(ptr[0])
    # # if current node isn't the goal state->continue exploring
    # while not problem.isGoalState(ptr[0]):
    #     # push all the unexplored successors of current node onto queue
    #     for i in problem.getSuccessors(ptr[0]):
    #         if(i[0] not in explored):
    #             explored.add(i[0])
    #             queue.push(i)
    #             if(i not in res_dict.keys()):
    #                 res_dict[i]=ptr #相当于记录父节点
    #     ptr=queue.pop()
    # res=[]
    # while(ptr[0]!=problem.getStartState()):
    #     res.append(ptr[1])
    #     ptr=res_dict[ptr]
    # res.reverse()
    # return res


def uniformCostSearch(problem: SearchProblem):
    # # 
    # another_dict={}
    # # the most cheapest path so far is in the front
    # queue=util.PriorityQueue()
    # # store explored nodes
    # explored=set()
    # # the start state
    # ptr=problem.getStartState()
    # # key: current node; value: parent node   store the parent node to a specific node
    # res_dict={} 
    # # add start state to explored set
    # explored.add(ptr)
    # another_dict[ptr]=((ptr,'Stop',0),0) # record the culminated score to this node so far
    # # if current node isn't explored, add it to the explored set
    # while not problem.isGoalState(ptr):
    #     explored.add(ptr)
    #     # add the current node's successors into explored set
    #     for i in problem.getSuccessors(ptr):
    #         if(i[0] not in explored):
    #             if(queue.update(i[0],i[2]+another_dict[ptr][1])):
    #                 another_dict[i[0]]=(i,i[2]+another_dict[ptr][1])
    #                 res_dict[i[0]]=ptr
    #     ptr=queue.pop()
    # res=[]
    # while(ptr!=problem.getStartState()):
    #     res.append(another_dict[ptr][0][1])
    #     ptr=res_dict[ptr]
    # res.reverse()
    # print(res)
    # return res
    """Search the node that has the lowest combined cost and heuristic first."""
    import copy
    fringe = util.PriorityQueue()
    ptr = (problem.getStartState(),[])
    explored = []
    fringe.push(ptr,0)
    #explored.append(ptr[0])
    while not fringe.isEmpty():
        ptr = fringe.pop()
        if problem.isGoalState(ptr[0]):
            return ptr[1]
        if ptr[0] not in explored:
            explored.append(ptr[0])
            for i in problem.getSuccessors(ptr[0]):
                # if i[0] not in explored:
                #     explored.append(i[0])
                successor_path = copy.deepcopy(ptr[1])
                successor_path.append(i[1])
                cost = problem.getCostOfActions(successor_path)
                fringe.update((i[0],successor_path),cost)
    return ptr[1]
    

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

    
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    import copy
    fringe = util.PriorityQueue()
    ptr = (problem.getStartState(),[])
    explored = []
    fringe.push(ptr,heuristic(problem.getStartState(),problem))
    #explored.append(ptr[0])
    while not fringe.isEmpty():
        ptr = fringe.pop()
        if problem.isGoalState(ptr[0]):
            return ptr[1]
        if ptr[0] not in explored:
            explored.append(ptr[0])
            for i in problem.getSuccessors(ptr[0]):
                # if i[0] not in explored:
                #     explored.append(i[0])
                successor_path = copy.deepcopy(ptr[1])
                successor_path.append(i[1])
                cost = heuristic(i[0],problem)+problem.getCostOfActions(successor_path)
                fringe.update((i[0],successor_path),cost)
    return ptr[1]
    
    # another_dict=util.Counter()
    # queue=util.PriorityQueue()
    # explored=[]
    # ptr=problem.getStartState()
    # res_dict=util.Counter()
    # explored.append(ptr)
    # another_dict[ptr]=((ptr,'Stop',0),heuristic(ptr,problem))
    # while not problem.isGoalState(ptr):
    #     explored.append(ptr)
    #     for i in problem.getSuccessors(ptr):
    #         if(i[0] not in explored):
    #             dist=i[2]+another_dict[ptr][1]+heuristic(i[0],problem)-heuristic(another_dict[ptr][0][0],problem)
    #             if(queue.update(i[0],dist)):
    #                 another_dict[i[0]]=(i,dist)
    #                 res_dict[i[0]]=ptr
    #     ptr=queue.pop()
    # res=[]
    # while(ptr!=problem.getStartState()):
    #     res.append(another_dict[ptr][0][1])
    #     ptr=res_dict[ptr]
    # res.reverse()
    # print(res)
    # return res

    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
