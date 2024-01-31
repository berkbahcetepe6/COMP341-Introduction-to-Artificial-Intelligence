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

        "*** YOUR CODE HERE ***"

        foodLoc = newFood.asList()

        if len(foodLoc) > 0:
            foodDist = [manhattanDistance(newPos, food) for food in foodLoc]  #stores the distances to each food]
            foodCount = len(foodDist)  #counts the number of foods left in order to avoid Pacman to stop or move back and forth next to the closest food if the ghost is not close to Pacman
        else: 
            foodCount = 0
            foodDist = [0]

        if newScaredTimes[0] == 0:
            ghostsToRun = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        else:
            ghostsToRun = [0 for _ in range(len(newGhostStates))]

        stopPenalty = 1 if action == Directions.STOP else 0 #penalty if Pacman stops
            
        return successorGameState.getScore() - min(foodDist) + min(ghostsToRun) - 5 * stopPenalty - foodCount  #the weight of stopPenalty is determined by trial and error

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
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
        agentIndex = 0
        return self.value(gameState, agentIndex, self.depth)[1]
        #util.raiseNotDefined()


    def value(self, gameState: GameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.min_value(gameState, agentIndex, depth)

    def max_value(self, gameState: GameState, agentIndex, depth):
        
        legalActions = gameState.getLegalActions(agentIndex)
        v, action = float("-inf"), Directions.STOP

        for actions in legalActions:
            successors = gameState.generateSuccessor(agentIndex, actions)
            new_v = self.value(successors, 0, depth - 1)[0] if agentIndex == gameState.getNumAgents() - 1 else self.value(successors, agentIndex + 1, depth)[0]
            if new_v > v:
                v = new_v
                action = actions

        return v, action

    def min_value(self, gameState: GameState, agentIndex, depth):

        legalActions = gameState.getLegalActions(agentIndex)
        v, action = float("inf"), Directions.STOP

        for actions in legalActions:
            successors = gameState.generateSuccessor(agentIndex, actions)
            new_v = self.value(successors, 0, depth - 1)[0] if agentIndex == gameState.getNumAgents() - 1 else self.value(successors, agentIndex + 1, depth)[0]
            if new_v < v:
                v = new_v
                action = actions
                    
        return v, action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        agentIndex = 0
        return self.value(gameState, agentIndex, self.depth)[1]
        #util.raiseNotDefined()


    def value(self, gameState: GameState, agentIndex, depth, alpha= float("-inf"), beta= float("inf")):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)

    def max_value(self, gameState: GameState, agentIndex, depth, alpha= float("-inf"), beta= float("inf")):
        
        legalActions = gameState.getLegalActions(agentIndex)
        v, action = float("-inf"), Directions.STOP

        for actions in legalActions:
            successors = gameState.generateSuccessor(agentIndex, actions)
            new_v = self.value(successors, 0, depth - 1, alpha, beta)[0] if agentIndex == gameState.getNumAgents() - 1 else self.value(successors, agentIndex + 1, depth, alpha, beta)[0]
            if new_v > v:
                v = new_v
                action = actions
            if v > beta:
                return v, action
            alpha = max(alpha, v)
            
        return v, action

    def min_value(self, gameState: GameState, agentIndex, depth, alpha= float("-inf"), beta= float("inf")):

        legalActions = gameState.getLegalActions(agentIndex)
        v, action = float("inf"), Directions.STOP

        for actions in legalActions:
            successors = gameState.generateSuccessor(agentIndex, actions)
            new_v = self.value(successors, 0, depth - 1, alpha, beta)[0] if agentIndex == gameState.getNumAgents() - 1 else self.value(successors, agentIndex + 1, depth, alpha, beta)[0]
            if new_v < v:
                v = new_v
                action = actions
            if v < alpha:
                return v, action
            beta = min(beta, v)                   
        return v, action
    
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
        agentIndex = 0
        return self.value(gameState, agentIndex, self.depth)[1]
        #util.raiseNotDefined()


    def value(self, gameState: GameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.exp_value(gameState, agentIndex, depth)

    def max_value(self, gameState: GameState, agentIndex, depth):
        
        legalActions = gameState.getLegalActions(agentIndex)
        v, action = float("-inf"), Directions.STOP

        for actions in legalActions:
            successors = gameState.generateSuccessor(agentIndex, actions)
            new_v = self.value(successors, 0, depth - 1)[0] if agentIndex == gameState.getNumAgents() - 1 else self.value(successors, agentIndex + 1, depth)[0]
            if new_v > v:
                v = new_v
                action = actions

        return v, action 
    
        return self.value(gameState, 0, self.depth)[1]
    
    def exp_value(self, gameState: GameState, agentIndex, depth):

        v = []
        legalActions = gameState.getLegalActions(agentIndex)
        for actions in legalActions:
            successors = gameState.generateSuccessor(agentIndex, actions)
            v.append(self.value(successors, 0, depth - 1)[0] if agentIndex == gameState.getNumAgents() - 1 else self.value(successors, agentIndex + 1, depth)[0])
        return sum(v)/len(legalActions), Directions.STOP
    
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Uses minimum food and ghost distances, number of capsules available as features. If ghosts are edible, I tried to minimize the minimum distance, whereas if we try to escape from them I tried to maximize the minimum distance
    "*** YOUR CODE HERE ***"
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsuleLocs = currentGameState.getCapsules()

    score = currentGameState.getScore()
    foodLoc = newFood.asList()

    if len(foodLoc) > 0:
        score += 2 / min([manhattanDistance(newPos, food) for food in foodLoc])  #stores the distances to each food
        #foodCount = len(foodDist)  #counts the number of foods left in order to avoid Pacman to stop or move back and forth next to the closest food if the ghost is not close to Pacman

    for ghost in newGhostStates:
        ghostDist = min([manhattanDistance(newPos, ghost.getPosition())])
        if ghostDist != 0:
            score += -1.5 / ghostDist if newScaredTimes[0] == 0 else 5 / ghostDist
    return score - len(capsuleLocs)

# Abbreviation
better = betterEvaluationFunction
