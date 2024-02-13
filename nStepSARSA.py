"""
Description
===========
The purpose of this script is to implement n-Step Bootstrapping SARSA algorithm to the CartPole environment from gymnasium
"""
import numpy as np
import gymnasium as gym
import time
import pickle

class nStepSARSA:
    def __init__(self, env, initialPolicy = None, nSteps = 2, discountFactor = 1, stepSize = 0.2, epsilon = 0.2, maxIterations = 1000):
        """_summary_

        Args:
            env (Object): Gymnasium Environment
            initialPolicy (ndarray): User can provide initial policy to start. If None the program will generate its own random initial policy. Defaults to None.
            nSteps (int): Steps of bootstrapping. Defaults to 2.
            discountFactor (float): Discount factor for future rewards. Defaults to 0.9.
            stepSize (float): Update rule step size. Defaults to 0.1.
            epsilon (float): For exploration and exploitation. Defaults to 0.1.
            maxIterations (int): Max number of Iterations of Generalized Policy Iterations. Defaults to 1000.
        """

        self.env            = env
        self.n              = nSteps
        self.gamma          = discountFactor
        self.alpha          = stepSize
        self.epsilon        = epsilon
        self.maxIterations  = maxIterations
        self.numBins        = 20 # for now there will be 20 bins for all the states, can be changed
        self.statesRanges   = np.array([[-2.5, 2.5],
                                        [-3,     3],
                                        [-0.26,   0.26],
                                        [-3,     3]])

        # calculate bins for Quantization
        self.statesBins = np.zeros((self.env.observation_space.shape[0], self.numBins))
        for i in range(self.env.observation_space.shape[0]):
            self.statesBins[i] = np.linspace(self.statesRanges[i, 0], self.statesRanges[i, 1], self.numBins)

        # max state number achieved by the Quantizer
        self.maxStateNum = self.StateQuantizer(self.env.observation_space.high)

        # policy
        # with open('PolicyCartPole.pkl', 'rb') as f:
        #     self.policy = pickle.load(f)
        if initialPolicy == None:
            self.policy = np.random.rand(self.maxStateNum, self.env.action_space.n)
            # normalizing the policy so that the sum is one
            self.policy = self.policy / np.sum(self.policy, axis = 1)[:, np.newaxis]
        else:
            self.policy = initialPolicy

        # Q value function and optimal policy initialization
        self.qValueFunction = np.zeros((self.maxStateNum, self.env.action_space.n))
        self.optimalPolicy  = np.zeros((self.maxStateNum, self.env.action_space.n))


    def StateQuantizer(self, state) -> list:
        """This function will convert the continuous observation state into discrete observation states.
        Plus it reduces the observation from 4 to 1, this is done to simplify things.

        Args:
            state (_type_): state in form of tuple

        Returns:
            list: returns the clipped and than quantized state
        """
        clippedState = np.zeros((len(state)))
        quantizedState = np.zeros((len(state)))
        for i in range(len(state)):
            clippedState[i] = np.clip(state[i], self.statesRanges[i, 0], self.statesRanges[i, 1])
            quantizedState[i] = np.digitize(clippedState[i], self.statesBins[i], right=False)
        return int( (quantizedState[0] + quantizedState[1] * self.numBins) + (quantizedState[2] + quantizedState[3] * self.numBins)  * (self.numBins ** 2 + self.numBins) )
        # return quantizedState

    def PolicyEvaluation(self) -> list:
        """
        This function performs policy evaluation for a given policy. For n-step bootstrapping,
        some extra steps needed to be perform even after termination condition is met to cater
        last visited states before the termination or truncation flag is rased. We will denote
        that with post termination flag variable.

        Returns:
            list: Q value function estimated for a policy
        """
        # timing parameters
        t = 0 # the original time of iterations
        tau = 0 # the tau of updation of parameters

        # total number of actions
        totalActions = np.arange(self.env.action_space.n)

        # storage containers
        listOfRewards = np.zeros((self.n), dtype=int)
        listOfStates = np.zeros((self.n), dtype=int)
        listOfActions = np.zeros((self.n), dtype=int)
        listOfGammas = np.zeros((self.n))
        for i in range(len(listOfGammas)):
            listOfGammas[i] = self.gamma ** i

        # states visited: this will help us to update the policy for these states only (will help in greedification)
        statesVisitedPerEpisode = []
        
        terminated = False
        truncated = False

        # for t = 0
        state, info = self.env.reset()
        state = self.StateQuantizer(state)
        action = np.random.choice(totalActions, p = self.policy[state])

        # for t > 0
        while True:
            t = t + 1
            tau = t - self.n 
            statesVisitedPerEpisode.append(state)
            
            # excite the environment before terminate or truncate
            if terminated == False and truncated == False:
                nextState, reward, terminated, truncated, _ = self.env.step(action)
                nextState = self.StateQuantizer(nextState)
                nextAction = np.random.choice(totalActions, p = self.policy[nextState])

                # store the rewards
                listOfRewards = np.roll(listOfRewards, -1)
                listOfRewards[self.n - 1] = reward
                listOfStates = np.roll(listOfStates, -1)
                listOfStates[self.n - 1] = state
                listOfActions = np.roll(listOfActions, -1)
                listOfActions[self.n - 1] = action

                if tau >= 0:
                    G = np.dot(listOfRewards, listOfGammas) + (self.gamma ** self.n) * self.qValueFunction[nextState][nextAction]
                    self.qValueFunction[listOfStates[0], listOfActions[0]] += self.alpha * (G - self.qValueFunction[listOfStates[0], listOfActions[0]])

            else:
                # this block si for post terminal or truncation updation
                for i in range(self.n):
                    G = np.dot(listOfRewards[i : ], listOfGammas[0 : len(listOfRewards[i : ])])
                    self.qValueFunction[listOfStates[i], listOfActions[i]] += self.alpha * (G - self.qValueFunction[listOfStates[i], listOfActions[i]])
                break

            state = nextState
            action = nextAction
        return self.qValueFunction, statesVisitedPerEpisode

    def eGreedification(self, statesVisitedPerEpisode) -> list:
        """
        This function outputs an e-soft policy based on the qValueFunction and states visited in the episode
        Args:
            statesVisitedPerEpisode : the list of states that are visited in the epsiode
        Returns:
            list: Updated policy
        """
        for state in statesVisitedPerEpisode:
            # print(state)
            self.policy[state, :] =  self.epsilon / self.env.action_space.n
            maximumPoints = np.where(self.qValueFunction[state, : ] ==  np.max(self.qValueFunction[state, :]))
            self.policy[state, maximumPoints[0]] = (1 - self.epsilon) / len(maximumPoints[0]) + self.epsilon / self.env.action_space.n
        return self.policy

    def PolicyIteration(self) -> list:
        """
        This function gives the optimal or sub-optimal policy for the environment

        Returns:
            list: suboptimal policy
        """
        for i in range(self.maxIterations):
            _, statesVistiedPerEpisode = self.PolicyEvaluation()
            self.eGreedification(statesVistiedPerEpisode)
            if i % 50 == 0:
                print(f'num episode : {i} , length of episode : {len(statesVistiedPerEpisode)} ')
        
        # optimal policy
        for i in range(self.maxStateNum):
            maximumPoints = np.where(self.qValueFunction[i, :] == np.max(self.qValueFunction[i, :]))[0]
            self.optimalPolicy[i, maximumPoints] = 1/len(maximumPoints)        
        with open('PolicyCartPole.pkl', 'wb') as f:
            pickle.dump(self.optimalPolicy, f)
        f.close()
        return self.optimalPolicy

    def TestPolicy(self):
        env = gym.make('CartPole-v1', render_mode = 'human', max_episode_steps=2000)
        with open('PolicyCartPole.pkl', 'rb') as f:
            self.policy = pickle.load(f)
        f.close()
        state, _ = env.reset()
        state = self.StateQuantizer(state)
        totalActions = np.arange(env.action_space.n)
        cumReward = 0
        terminated = False
        truncated = False
        while True:
            action = np.random.choice(totalActions, p = self.policy[state])
            state, reward, terminated, truncated, info = env.step(action)
            state = self.StateQuantizer(state)
            cumReward += reward
            if terminated or truncated:
                break
        time.sleep(1)
        env.close()
        return cumReward




if __name__ == '__main__':
    env = gym.make('CartPole-v1', max_episode_steps=2000)
    obj = nStepSARSA(env, maxIterations = 10000)
    print(obj.TestPolicy())
    obj.PolicyIteration()
    print(obj.TestPolicy())
    print(obj.qValueFunction.max())
    
    