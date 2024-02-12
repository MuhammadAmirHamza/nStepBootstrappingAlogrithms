"""
Description
===========
The purpose of this script is to implement n-Step Bootstrapping SARSA algorithm to the CartPole environment from gymnasium
"""
import numpy as np
import gymnasium as gym
import time

class nStepSARSA:
    def __init__(self, env, initialPolicy = None, nSteps = 2, discountFactor = 0.9, stepSize = 0.1, epsilon = 0.1, maxIterations = 1000):
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
        
        self.env = env
        self.n = nSteps
        self.gamma = discountFactor
        self.alpha = stepSize
        self.epsilon = epsilon
        self.maxIterations = maxIterations
        self.numBins = 40 # for now there will be 20 bins for all the states, can be changed
        self.statesRanges = np.array([[-2.4, 2.4],
                                    [-3,     3],
                                    [-2.1,   2.1],
                                    [-3,     3]])
        
        # calculate bins for Quantization
        self.statesBins = np.zeros((self.env.observation_space.shape[0], self.numBins))
        for i in range(self.env.observation_space.shape[0]):
            self.statesBins[i] = np.linspace(self.statesRanges[i, 0], self.statesRanges[i, 1], self.numBins)
        
        # max state number achieved by the Quantizer
        maxLin, maxAng = self.StateQuantizer(self.env.observation_space.high)
        
        # policy   
        if initialPolicy == None:
            self.policy = np.random.rand(maxLin, maxAng, self.env.action_space.n)
            self.policy = self.policy / np.expand_dims(np.sum(self.policy, axis = 2), axis = 2)
        else:
            pass
    
    
    def StateQuantizer(self, state) -> list:
        """This function will convert the continuous observation state into discrete observation states.
        Plus it reduces the observation from 4 to 2, this is done to simplify things

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
            print(type(quantizedState[i]))
        return [int(quantizedState[0] + quantizedState[1] * self.numBins), int(quantizedState[2] + quantizedState[3] * self.numBins) ]
        # return quantizedState
    
    def PolicyEvaluation(self) -> list:
        """
        This function performs policy evaluation for a given policy

        Returns:
            list: Q value function estimated for a policy
        """
        pass
    
    def eGreedification(self) -> list:
        """
        This function outputs an e-soft policy based on the qValueFunction

        Returns:
            list: Updated policy
        """
        pass
    
    def PolicyIteration(self) -> list:
        """
        This function gives the optimal or sub-optimal policy for the environment

        Returns:
            list: suboptimal policy
        """
        pass
    
    
if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode = 'human')
    obj = nStepSARSA(env)
    state, info = env.reset()
    print(obj.policy)