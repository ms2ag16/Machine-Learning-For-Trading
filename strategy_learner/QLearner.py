# Mi Sun  msun85
"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states=num_states
        self.alpha=alpha
        self.gamma=gamma
        self.rar=rar
        self.radr=radr
        self.dyna=dyna

        # initial Q table
        self.Q=np.random.uniform(-1,1,size=(self.num_states,self.num_actions))
        self.D = []

        # initial dyna
        if (self.dyna!=0):
            self.Tcount=np.ndarray(shape=(self.num_states, self.num_actions, self.num_states))
            # initial Tcount with 0.00001
            self.Tcount.fill(0.00001)
            # learning T is the probability
            self.T=self.Tcount/self.Tcount.sum(axis=2,keepdims=True)
            # learning R
            self.R=np.ndarray(shape=(self.num_states, self.num_actions))
            self.R.fill(-1.0)



    def author(self):
        return 'msun85'

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if(rand.random()<self.rar):
            rand.seed(1)
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s,:])
            self.rar *= self.radr

        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        action=np.argmax(self.Q[s_prime,:])
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] \
                                 + self.alpha * (r + self.gamma * np.max(self.Q[s_prime,action]))


        # checks to see if we're doing dyna.
        count=self.dyna
        self.D.append((self.s,self.a,s_prime,r))
        while(count):
            index=rand.randint(0,len(self.D)-1)
            s_current,a_current,s_next,r_next=self.D[index]
            a_next=np.argmax(self.Q[s_next,:])
            self.Q[s_current,a_current]=(1 - self.alpha) * self.Q[s_current,a_current] \
                                 + self.alpha * (r_next + self.gamma * self.Q[s_next,a_next])
            count-=1

        action = np.argmax(self.Q[s_prime, :])
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
            self.rar = self.rar * self.radr

        self.s=s_prime
        self.a=action

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"