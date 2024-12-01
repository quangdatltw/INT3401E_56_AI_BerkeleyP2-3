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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()

        # Tinh khoang cach Manhattan den thuc an gan nhat
        foodDistances = [manhattanDistance(newPos, food) for food in newFood]
        if foodDistances:
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 0

        # Tinh khoang cach Manhattan den ma gan nhat
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances)

        # Neu gap ma, tra ve diem am
        if minGhostDistance <= 1:
            return -9999

        # Neu dung lai, tra ve diem am (Tranh bi dung yen cho)
        if action == Directions.STOP:
            return -9999

        # Neu di nguoc huong, tra ve diem am (Tranh di lap trai-phai, len-xuong)
        reverse = Directions.REVERSE[currentGameState.getPacmanState().configuration.direction]
        if action == reverse:
            return -5000

        # Tra ve diem moi + 1/(khoang cach den thuc an gan nhat + 1) ( + 1 de tranh chia cho 0)
        return successorGameState.getScore() + 1.0 / (minFoodDistance + 1)

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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        numberOfGhosts = gameState.getNumAgents() - 1

        #Used only for pacman agent hence agentindex is always 0.
        def maxLevel(gameState,depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,minLevel(successor,currDepth,1))
            return maxvalue
        
        #For all ghosts.
        def minLevel(gameState,depth, agentIndex):
            minvalue = 999999
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minvalue = min (minvalue,maxLevel(successor,depth))
                else:
                    minvalue = min(minvalue,minLevel(successor,depth,agentIndex+1))
            return minvalue
        
        #Root level action.
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a min level. Hence calling min for successors of the root.
            score = minLevel(nextState,0,1)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        #Used only for pacman agent hence agentindex is always 0.
        def maxLevel(gameState,depth,alpha, beta):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,minLevel(successor,currDepth,1,alpha1,beta))
                if maxvalue > beta:
                    return maxvalue
                alpha1 = max(alpha1,maxvalue)
            return maxvalue
        
        #Cho ma
        def minLevel(gameState,depth,agentIndex,alpha,beta):
            minvalue = 999999
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            beta1 = beta
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents()-1):
                    minvalue = min (minvalue,maxLevel(successor,depth,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
                else:
                    minvalue = min(minvalue,minLevel(successor,depth,agentIndex+1,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
            return minvalue

        # Alpha-Beta Pruning
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        alpha = -999999
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a min level. Hence calling min for successors of the root.
            score = minLevel(nextState,0,1,alpha,beta)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
            # Updating alpha value at root
            alpha = max(alpha,score)
        return returnAction

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
        # util.raiseNotDefined()

        # De quy cho Pacman
        def maxLevel(gameState, depth):
            currDepth = depth + 1
            # Dung neu state la win/lose hoac da dat den do sau lon nhat.
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)

            # Duyet qua cac hanh dong de lay gia tri tot nhat.
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                maxvalue = max(maxvalue, expectLevel(successor, currDepth, 1))
            return maxvalue

        # De quy cho Ghost.
        def expectLevel(gameState, depth, agentIndex):
            # Dung neu state la win/lose.
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)

                # Neu la ghost cuoi goi den ham maxLevel de xu li cho Pacman.
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxLevel(successor, depth)
                # Tiep tuc xu li Ghost.
                else:
                    expectedvalue = expectLevel(successor, depth, agentIndex + 1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if numberofactions == 0:
                return 0

            # Chia gia tri ki vong trung binh.
            return float(totalexpectedvalue) / float(numberofactions)

        # Init o root.
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)

            # Sau root la tang expect nen goi ham expectLevel
            score = expectLevel(nextState, 0, 1)

            # Cap nhat score va action.
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    newPos = currentGameState.getPacmanPosition() # Vi tri hien tai
    newFood = currentGameState.getFood() # Cac food hien tai
    newGhostStates = currentGameState.getGhostStates() # Trang thai hien tai cua Ghost
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # Scared time cua ghost

    # Tinh khoang cach tu vi tri hien tai den cac food
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos, pos))

    # Tinh khoang cach tu vi tri hien tai den cac Ghost
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos, pos))

    # So luong capsule
    numberofPowerPellets = len(currentGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum(ghostDistance)
    reciprocalfoodDistance = 0 # Nghich dao tong khoang cach den food
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)

    # Cong diem: cang gan food cang duoc nhieu diem, cang an duoc nhieu cang duoc nhieu diem
    score += currentGameState.getScore() + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0: # Khi Ghost bi scared
        # Cong them diem neu Ghost dang so, tru diem neu con capsule, tru diem neu qua xa Ghost
        score += sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else: # Ghost binh thuong
        # Cong diem neu dung xa Ghost, cong diem neu con capsule
        score += sumGhostDistance + numberofPowerPellets
    return score

# Abbreviation
better = betterEvaluationFunction
