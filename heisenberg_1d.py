"""
heisenber_classical with molecular dynamics
"""


from functools import wraps
import numpy as np
from .measurements import Environment
from numpy.random import rand



class Heisenberg_1d(Environment):
    def __init__(self, N, T):
        self.energy_list = []
        self.magnetisation_list = []
        self.iteration = 0
        self.accepting = 0
        self.N = N
        self.T = T
        self.beta = 1 / T
        self.leap_steps = 40
        self.epsilon = 0.01
        self.config = self.initialstate()
        self.new_config = np.zeros((self.N, self.N,2))
        self.energy_value = self.energy(self.config)
        self.new_energy=0.
        self.dE_list= []



    def initialstate(self):
        config = 2*np.random.rand(self.N, self.N)*np.pi
        return config

    def dHdq(self, q, n):
        dHdq = np.zeros((self.N, self.N))                # q = [theta,phi]
        dHdq += -n[...,0]* np.sin(q)+n[...,1] * np.cos(q)
        return -dHdq*self.beta



    def change_one_site(self):
        self.a = np.random.randint(0, self.N)
        self.b = np.random.randint(0, self.N)
        self.new_config = self.config.copy()
        self.new_config[self.a,self.b] = 2 * np.random.rand() * np.pi
        # use update_energy
        self.new_energy = self.energy(self.new_config)


    def u_v_v(self, q):
        n = np.zeros((self.N, self.N, 2 ))
        n[...,0] += np.cos(q)
        n[...,1] += np.sin(q)

        return n

    def nb(self,config):
        nb= (
            self.u_v_v(np.roll(config,1 , axis = 0))
            + self.u_v_v(np.roll(config, 1,  axis=1))
        )
        return nb/2

    def energy(self, config, **params):
        cost = np.sum(self.u_v_v(config)[:,:] * self.nb(config)[:,:])
        return -cost

    """
    def update_energy(self, **params):
        nb = (
            self.u_v_v(np.roll(self.config,1 , axis = 0))
            + self.u_v_v(np.roll(self.config, -1, axis = 0))
            + self.u_v_v(np.roll(self.config, 1,  axis=1))
            + self.u_v_v(np.roll(self.config, -1,  axis=1))
        )
        self.config = self.leap_frog(
            self.config, np.random.normal(0.0, 1.0, self.config.shape), nb
        )
        cost = np.sum(self.u_v_v(self.config)[:,:] * nb[:,:])#[a, b]), nb)
        return cost/4
    """

    def stop(self,**params):
        return self.energy_list, self.dE_list, self.accepting / params["max_iterations"]


    def observables(self, **params):
        if params["hmc"]:
            self.energy_list.append(self.energy_value)
            self.magnetisation_list.append(self.magnetisation())
        else:
            if self.iteration % self.N ** 2 == 0:
                self.energy_list.append(self.energy_value)
                self.magnetisation_list.append(self.magnetisation())

    def leap_frog(self, q, p):
        for i in range(self.leap_steps):
            p = p - self.dHdq(q, self.nb(q)) * self.epsilon / 2
            q = q + p * self.epsilon
            p = p - self.dHdq(q, self.nb(q)) * self.epsilon / 2
        return q,p






#env = Heisenberg_1d(N=16, T=1.53)
#metropolis(env,hmc=True, max_iterations=5)
