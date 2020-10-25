# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, epsillon=0.5, alpha=0.2, gamma=0.8, **args):
        CaptureAgent.__init__(self, index)
        self.epsillon = epsillon
        self.alpha = alpha
        self.discout = gamma
        self.lastState = None
        self.lastAction = None
        self.targetPos = None
        self.mazeSize = None
        self.specificPath = []
 
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        walls = gameState.getWalls()
        self.mazeSize = walls.height * walls.width
        
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def doAction(self, gameState, action):
        """
        update last state and action
        """
        self.lastState = gameState
        self.lastAction = action


    def getQValue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        return features * self.weights
    
    def update(self, gameState, action, nextState, reward):
        actions = nextState.getLegalActions(self.index)
        actions.remove('Stop')
        values = [self.getQValue(nextState, a) for a in actions]
        maxValue = max(values)
        features = self.getFeatures(gameState, action)
        
        diff = (reward + self.discout * maxValue) - self.getQValue(gameState, action)

        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]

    def getGhosts(self, gameState):
        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition()]
        return ghosts

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]
        return invaders
    
    def getSafeActions(self, gameState, border=None):
        if border == None:
            border = self.border

        safeActions = []
        myPos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            myNextPos = successor.getAgentPosition(self.index)

            finalNode = self.aStarSearch(successor, border, [myPos])
            if finalNode[2] < self.mazeSize:
                safeActions.append(action)
        
        return safeActions

    def getAlternativePath(self, gameState, minPathLength=5, penaltyDist=2, exploreRange=5):
        walls = gameState.getWalls()
        myPos = gameState.getAgentPosition(self.index)
        ghosts = self.getGhosts(gameState)
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        targetList = foodList + capsuleList
        
        penaltyPos = []
        
        for ghost in ghosts:
            for x in range(max(1, myPos[0] - exploreRange), min(myPos[0] + exploreRange, walls.width)):
                for y in range(max(1, myPos[1] - exploreRange), min(myPos[1] + exploreRange, walls.height)):
                    pos = (int(x), int(y))
                    if not pos in walls.asList():
                        distToGhost = self.getMazeDistance(pos, ghost.getPosition())
                        if distToGhost <= penaltyDist:
                            penaltyPos.append(pos)
                    if pos in targetList:
                        targetList.remove(pos)

        if len(targetList) == 0:
            return [], None

        finalNode = self.aStarSearch(gameState, targetList, penaltyPos)
        pathLength = min(minPathLength, len(finalNode[1]))
        return finalNode[1][0:pathLength], finalNode[0]
                        
    def aStarSearch(self, gameState, goals, penaltyPos=[], avoidGhost=True):
        walls = gameState.getWalls().asList()
        ghosts = self.getGhosts(gameState)
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        startPos = gameState.getAgentPosition(self.index)
        currentNode = (startPos, [], 0)
        pQueue = util.PriorityQueueWithFunction(lambda item: item[2] + min(self.getMazeDistance(item[0], goal) for goal in goals))
        pQueue.push(currentNode)
        closed = set()

        while currentNode[0] not in goals and not pQueue.isEmpty():
            currentNode = pQueue.pop()
            successors = [((currentNode[0][0] + v[0], currentNode[0][1] + v[1]), a) for v, a in zip(actionVectors ,actions)]
            legalSuccessors = [s for s in successors if s[0] not in walls]

            for successor in legalSuccessors:
                if successor[0] not in closed:
                    closed.add(successor[0])

                    position = successor[0]
                    path = currentNode[1] + [successor[1]]
                    cost = currentNode[2] + 1
                    wallCount = 0

                    if successor[0] in penaltyPos:
                        cost += self.mazeSize

                    if avoidGhost:
                        distToGhost = min([self.getMazeDistance(successor[0], a.getPosition()) for a in ghosts])
                        if distToGhost > 0:
                            cost += (self.mazeSize / 4) / distToGhost

                    pQueue.push((position, path, cost))

        return currentNode
    
    def isOppoentsScared(self, gameState, timer=4):
        myPos = gameState.getAgentPosition(self.index)
        ghosts = self.getGhosts(gameState)
        
        if len(ghosts) == 0:
            return False
        
        closestGhost = min(ghosts, key=lambda x: self.getMazeDistance(myPos, x.getPosition()))
        
        return closestGhost.scaredTimer > timer

    def isStucking(self, gameState, stuckingCount=4):
        history = self.observationHistory
        count = 0
        myPos = gameState.getAgentPosition(self.index)
        
        if len(history) > 0:
            for i in range(min(10, len(history))):
                myPastPos = history[-i - 1].getAgentPosition(self.index)
                if myPastPos == myPos:
                    count += 1
            
        return count >= stuckingCount

    def isChased(self, gameState, chasedCount=3, minDist=3):
        history = self.observationHistory
        myState = gameState.getAgentState(self.index)
        ghosts = self.getGhosts(gameState)
        
        if len(history) == 0 or len(ghosts) == 0 or not myState.isPacman:
            return False

        myPos = myState.getPosition()
        distToGhost = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
        
        if distToGhost > minDist:
            return False

        for i in range(min(chasedCount, len(history))):
            pastState = history[-i - 1]
            myPastPos = pastState.getAgentPosition(self.index)
            pastGhosts = self.getGhosts(pastState)
            if len(pastGhosts) == 0:
                return False
            
            pastDistToGhost = min([self.getMazeDistance(myPastPos, a.getPosition()) for a in pastGhosts])
 
            if pastDistToGhost != distToGhost:
                return False
            
        return True

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def __init__(self, index, **args):
        ReflexCaptureAgent.__init__(self, index, **args)
        self.weights = util.Counter({
            'bias': 1.0,
            'distToTarget': -10.0,
            'distToGhost': 5.0,
            'distToBorder': -1.0,
            'eatFood': 1.0,
            '#-of-ghosts-2-step-away': -1,
        })
        self.border = None
    

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        
        walls = gameState.getWalls()
        border = []
        x = walls.width // 2
        if self.red:
            x -= 1

        for y in range(1, walls.height - 1):
            if not walls[x][y] and (x, y) != self.start:
                border.append((x, y))
        self.border = border

        myPos = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()
        if len(foodList) > 0:
            self.targetPos = max(foodList, key=lambda x: self.getMazeDistance(myPos, x))
        
    
    def observationFunction(self, gameState):
        """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
        """
        if self.lastState:
            myPos = gameState.getAgentPosition(self.index)
            if myPos == self.start:
                self.specificPath = []
                foodList = self.getFood(gameState).asList()
                if len(foodList) > 0:
                    self.targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))

            if len(self.specificPath) == 0:
                reward = self.getReward(gameState)
                self.update(self.lastState, self.lastAction, gameState, reward)

        return gameState

    def getFeatures(self, gameState, action):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        successor = self.getSuccessor(gameState, action)
        myNextState = successor.getAgentState(self.index)
        myNextPos = myNextState.getPosition()

        foodList = self.getFood(successor).asList()
        capsuleList = self.getCapsules(successor)
        ghosts = self.getGhosts(successor)
        closestBorder = min(self.border, key=lambda x: self.getMazeDistance(myNextPos, x))
        minDistToBorder = min([self.getMazeDistance(myNextPos, b) for b in self.border])

        # Update target position
        timeLeft = gameState.data.timeleft / gameState.getNumAgents()
        if timeLeft - minDistToBorder <= 10:
            self.targetPos = closestBorder

        if len(foodList) <= 2:
            self.targetPos = closestBorder

        if len(foodList) > 2:
            if myNextPos == self.targetPos or myNextPos in foodList:
                self.targetPos = min(foodList, key=lambda x: self.getMazeDistance(myNextPos, x))
            
        if len(ghosts) > 0 and len(foodList) > 2:
            minDistToFood =  min([self.getMazeDistance(myNextPos, f) for f in foodList])
            minDistToGhost =  min([self.getMazeDistance(myNextPos, a.getPosition()) for a in ghosts])

            if not self.isOppoentsScared(successor):
                if self.isChased(gameState):
                    self.targetPos = closestBorder

                if myState.numCarrying >= 3 and minDistToBorder < minDistToFood:
                    self.targetPos = closestBorder
                        
                if myState.numCarrying >= 5  and minDistToGhost <= 5:
                    self.targetPos = closestBorder
                
                if len(capsuleList) > 0:
                    minDistToCapsule =  min([self.getMazeDistance(myNextPos, c) for c in capsuleList])
                    
                    if self.isChased(successor) and minDistToCapsule < minDistToBorder:
                        self.targetPos = min(capsuleList, key=lambda x: self.getMazeDistance(myNextPos, x))
                   
    
        # Calculate features
        distToGhost = 0.0
        if len(ghosts) > 0:
            distToGhost = min([self.getMazeDistance(myNextPos, a.getPosition()) for a in ghosts])
            if not self.isOppoentsScared(successor) and myState.isPacman and myNextPos == self.start:
                distToGhost = -999999

        capsuleList = self.getCapsules(gameState)
        distToCapsule = 0.0
        if len(capsuleList) > 0:
            distToCapsule = min([self.getMazeDistance(myNextPos, capsule) for capsule in capsuleList])
        
        features = util.Counter()
        features['bias'] = 1.0
        features['distToTarget'] = self.getMazeDistance(myNextPos, self.targetPos) / self.mazeSize
        features['distToGhost'] = distToGhost / self.mazeSize
        features['distToCapsule'] = distToCapsule / self.mazeSize
        
        features['#-of-ghosts-2-step-away'] = len([ghost for ghost in ghosts if self.getMazeDistance(myNextPos, ghost.getPosition()) <= 2])
        
        if self.isOppoentsScared(successor):
            features['distToGhost'] = 0.0
            features['#-of-ghosts-2-step-away'] = 0.0

        foodList = self.getFood(gameState).asList()
        if not features['#-of-ghosts-2-step-away'] and myNextPos in foodList:
            features['eatFood'] = 1.0
        
        if myState.numCarrying > 0: 
            features['distToBorder'] = min([self.getMazeDistance(myPos, b) for b in self.border]) / (self.mazeSize / 4)

        return features


    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        ghosts = self.getGhosts(gameState)

        if self.lastState:
            myLastState = self.lastState.getAgentState(self.index)
        
            ghosts = self.getGhosts(gameState)
            if len(ghosts) > 0:
                minDistToGhost =  min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
                if not myState.isPacman and myLastState.isPacman and minDistToGhost <= 3 and self.specificPath == []:
                    path, target = self.getAlternativePath(gameState, minPathLength=5)
                    self.specificPath = path
                    self.targetPos = target
        
        if len(self.specificPath) > 0:
            return self.specificPath.pop(0)
        elif self.isStucking(gameState):
            actions, target = self.getAlternativePath(gameState, minPathLength=5)
            if len(actions) > 0:
                self.specificPath = actions
                self.targetPos = target
                return self.specificPath.pop(0)
            else:
                actions = gameState.getLegalActions(self.index)
                return random.choice(actions)

        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        if len(ghosts) > 0:
            distToGhost = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
            if not self.isOppoentsScared(gameState) and myState.isPacman and distToGhost <= 6:
                safeActions = self.getSafeActions(gameState)
                if len(safeActions) > 0:
                    actions = safeActions

        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        
        self.doAction(gameState, bestAction)

        return bestAction


    def getReward(self, gameState):
        reward = 0
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        myLastPos = self.lastState.getAgentPosition(self.index)
        foodList = self.getFood(self.lastState).asList()
        capsuleList = self.getCapsules(self.lastState)

        if myPos != self.targetPos:
            reward -= 1
        else:
            if myPos in foodList:
                reward += 1
            elif myPos in capsuleList:
                reward += 2
            else:
                reward += self.getScore(gameState) - self.getScore(self.lastState)

            distToPrevPos = self.getMazeDistance(myPos, myLastPos)        
            if distToPrevPos > 1:
                reward -= distToPrevPos / self.mazeSize

        return reward


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def __init__(self, index, **args):
        ReflexCaptureAgent.__init__(self, index, **args)
        self.weights = util.Counter({
            'bias': 1.0,
            'distToTarget': -10.0,
            'distToInvader': -1.0,
            'numOfInvaders': -1.0,
            'distToMissingFood': -1.0,
            'scaredScore': 1.0,
            'onDefense': 20.0,
        })
        self.initialFoodList = None
        self.border = None
        self.defenceBorder = None
        self.deepBorder = None
        self.opponentPacman = []
 
    
    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)

        self.initialFoodList = self.getFoodYouAreDefending(gameState).asList()

        walls = gameState.getWalls()
        self.mazeSize = walls.height * walls.width

        border = []
        x = walls.width // 2
        if self.red:
            x -= 1

        for y in range(1, walls.height - 1):
            if not walls[x][y] and (x, y) != self.start:
                border.append((x, y))
        self.border = border


        defenceBorder = []
        x = walls.width // 2
        if self.red:
            x -= 3
        else:
            x += 2

        for y in range(1, walls.height - 1):
            if not walls[x][y] and (x, y) != self.start:
                defenceBorder.append((x, y))
        self.defenceBorder = defenceBorder
        
        distCounter = util.Counter()
        for b in self.defenceBorder:
            dist = 0
            for food in self.getFoodYouAreDefending(gameState).asList():
                dist += self.getMazeDistance(b, food)
            distCounter[b] = dist

        self.targetPos = min(distCounter, key=distCounter.get)

        deepBorder = []
        x = walls.width // 2
        if self.red:
            x -= 5
        else:
            x += 4

        for y in range(1, walls.height - 1):
            if not walls[x][y] and (x, y) != self.start:
                deepBorder.append((x, y))
        self.deepBorder = deepBorder
        
        distCounter = util.Counter()
        for b in self.deepBorder:
            dist = 0
            for food in self.getFoodYouAreDefending(gameState).asList():
                dist += self.getMazeDistance(b, food)
            distCounter[b] = dist


    def observationFunction(self, gameState):
        """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
        """
        if self.lastState:
            myState = gameState.getAgentState(self.index)
            myPos = myState.getPosition()
            invaders = self.getInvaders(gameState)

            if not myState.isPacman:
                self.specificPath = []
                
            if myPos == self.start:
                self.specificPath = []
                foodList = self.getFood(gameState).asList()
                if len(invaders) > 0:
                    self.targetPos = min(invaders, key=lambda x: self.getMazeDistance(myPos, x.getPosition())).getPosition()
                else:
                    distCounter = util.Counter()
                    for b in self.defenceBorder:
                        dist = 0
                        for food in self.getFoodYouAreDefending(gameState).asList():
                            dist += self.getMazeDistance(b, food)
                        distCounter[b] = dist

                    self.targetPos = min(distCounter, key=distCounter.get)

            if len(self.specificPath) == 0:
                reward = self.getReward(gameState)
                self.update(self.lastState, self.lastAction, gameState, reward)

        return gameState


    def getFeatures(self, gameState, action):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        successor = self.getSuccessor(gameState, action)
        myNextState = successor.getAgentState(self.index)
        myNextPos = myNextState.getPosition()
        foodList = self.getFoodYouAreDefending(successor).asList()
        missingFoodList = self.getMissingFoods(successor)
        invaders = self.getInvaders(successor)
        ghosts = self.getGhosts(successor)

        # Update target position
        self.findOpponentPacman()

        if len(invaders) > 0:
            minDistToInvader = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])

            if minDistToInvader >= 8 and len(missingFoodList) > 0:
                minDist = float('inf')
                closestFoodFromMissing = None
                for missingFood in missingFoodList:
                    distFromMissingFood = min([self.getMazeDistance(missingFood, f) for f in foodList])
                    if distFromMissingFood < minDist:
                        closestFoodFromMissing = missingFood
                
                self.targetPos = closestFoodFromMissing
            else:
                self.targetPos = min(invaders, key=lambda x: self.getMazeDistance(myPos, x.getPosition())).getPosition()
        else:
            if len(self.opponentPacman) > 0:
                distCounter = util.Counter()
                for b in self.defenceBorder:
                    dist = 0
                    for p in self.opponentPacman:
                        distCounter[(b, p)] = self.getMazeDistance(b, gameState.getAgentPosition(p))

                self.targetPos = min(distCounter, key=distCounter.get)[0]
            else:
                distCounter = util.Counter()
                for b in self.defenceBorder:
                    dist = 0
                    for food in self.getFoodYouAreDefending(gameState).asList():
                        dist += self.getMazeDistance(b, food)
                    distCounter[b] = dist

                self.targetPos = min(distCounter, key=distCounter.get)

        if myNextState.scaredTimer > 0:
            if len(self.opponentPacman) > 0:
                agent = min(self.opponentPacman, key=lambda x: self.getMazeDistance(myPos, successor.getAgentPosition(x)))
                if not successor.getAgentState(agent).isPacman:
                    distCounter = util.Counter()
                    for b in self.deepBorder:
                        dist = 0
                        for p in self.opponentPacman:
                            distCounter[(b, p)] = self.getMazeDistance(b, gameState.getAgentPosition(p))

                    self.targetPos = min(distCounter, key=distCounter.get)[0]

                    

        # Calculate features
        distToInvader = 0.0
        if len(invaders) > 0:
            distToInvader = min([self.getMazeDistance(myNextPos, a.getPosition()) for a in invaders])

        features = util.Counter()
        features['bias'] = 1.0
        features['numOfInvaders'] = len(invaders) / 2.0
        features['distToTarget'] = self.getMazeDistance(myNextPos, self.targetPos) / self.mazeSize

        if myNextState.scaredTimer > 0:
            features['scaredScore'] = (distToInvader - myNextState.scaredTimer) / self.mazeSize

            if len(invaders) > 0:
                distToInvader = min([self.getMazeDistance(myNextPos, a.getPosition()) for a in invaders])
                if distToInvader <= 1:
                    features['distToInvader'] = 99999

        else:         
            features['scaredScore'] = 0.0
            features['distToInvader'] = distToInvader / self.mazeSize

        if not myNextState.isPacman:
            features['onDefense'] = 1.0
        else:
            features['onDefense'] = -1.0

        return features


    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        invaders = self.getInvaders(gameState)
        if myState.scaredTimer > 0 and len(invaders) > 0:
            distToGhost = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])
            if not myState.isPacman:
                safeActions = self.getSafeActions(gameState, self.defenceBorder)
                if len(safeActions) > 0:
                    actions = safeActions

        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        
        self.doAction(gameState, bestAction)

        return bestAction


    def getReward(self, gameState):
        reward = 0
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        myLastState = self.lastState.getAgentState(self.index)
        myLastPos = myLastState.getPosition()
        foodList = self.getFoodYouAreDefending(gameState).asList()
        lastFoodList = self.getFoodYouAreDefending(self.lastState).asList()
        capsuleList = self.getCapsulesYouAreDefending(gameState)
        lastCapsuleList = self.getCapsulesYouAreDefending(self.lastState)

        if myPos != self.targetPos:
            reward -= 1
        else:
            if len(foodList) < len(lastFoodList):
                reward -= 1
            elif len(capsuleList) < len(lastCapsuleList):
                reward -= 2
            else:
                reward += self.getScore(gameState) - self.getScore(self.lastState)
            
            distToPrevPos = self.getMazeDistance(myPos, myLastPos)
            if distToPrevPos > 1:
                reward -= distToPrevPos / self.mazeSize

        return reward

    def getMissingFoods(self, gameState, exploreRange=5):
        history = self.observationHistory
        for i in range(1, min(exploreRange, len(history))):
            foodList = self.getFoodYouAreDefending(history[-i]).asList()
            lastFoodLIst = self.getFoodYouAreDefending(history[-i - 1]).asList()

            missingList = [f for f in lastFoodLIst if f not in foodList]
            if len(missingList) > 0:
                return missingList
            else:
                return []

    def findOpponentPacman(self):
        opponentIndex = []
        if self.red:
            opponentIndex = [1, 3]
        else:
            opponentIndex = [0, 2]
        
        pacmanIndex = []
        history = self.observationHistory

        for index in opponentIndex:
            count = 0
            historyCount = 0
            
            for j in range(len(history)):
                pastState = history[-j - 1]
                historyCount += 1

                if pastState.getAgentState(index).isPacman:
                    count += 1
                    if count >= 5:
                        pacmanIndex.append(index)
                        break
                else:
                    count = 0
                    
        self.opponentPacman = pacmanIndex