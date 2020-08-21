"""
heisenber_classical with molecular dynamics
"""


from functools import wraps
import numpy as np
from .measurements import Environment
from numpy.random import rand



class Heisenberg_classical(Environment):
    def __init__(self, N, T):
        self.energy_list = []
        self.magnetisation_list = []
        self.iteration = 0
        self.accepting = 0
        self.N = N
        self.T = T
        self.beta = 1 / T
        self.tau = 1
        self.leap_steps = 50
        self.epsilon = self.tau/self.leap_steps
        self.config = self.initialstate()
        self.new_config = np.zeros((self.N, self.N,2))
        self.energy_value = self.energy(self.config)
        self.new_energy=0.

    def initialstate(self):
        config = np.zeros((self.N, self.N, 2))
        config[...,0] += np.arccos(2*np.random.rand(self.N, self.N)-1)
        config[...,1] += 2 * np.random.rand(self.N, self.N) * np.pi
        return config

    def dHdq(self, q, n):
        dHdq = np.zeros((self.N, self.N, 2))                # q = [theta,phi]
        dHdq[...,0] += -n[...,0]* np.cos(q[...,0])/np.sin(q[...,0]) * np.cos(q[...,1]) - n[...,1] * np.cos(q[...,0])/np.sin(q[...,0]) * np.sin(q[...,1]) + n[...,2]
        dHdq[...,1] += -n[...,0]* np.sin(q[...,0]) * np.sin(q[...,1]) + n[...,1] * np.sin(q[...,0]) *np.cos(q[...,1])
        #dHdq[...,0] += n[...,0]* np.cos(q[...,0])*np.cos(q[...,1])  + n[...,1] * np.cos(q[...,0]) * np.sin(q[...,1]) - n[...,2]*np.sin(q[...,0])
        #dHdq[...,1] += -n[...,0]* np.sin(q[...,0]) * np.sin(q[...,1]) + n[...,1] * np.sin(q[...,0]) *np.cos(q[...,1])
        return -dHdq*self.beta



    def change_one_site(self):
        self.a = np.random.randint(0, self.N)
        self.b = np.random.randint(0, self.N)
        self.new_config = self.config.copy()
        self.new_config[self.a,self.b,0] = np.arccos(2*np.random.rand()-1)
        self.new_config[self.a,self.b,1] = 2 * np.random.rand() * np.pi
        # use update_energy
        self.new_energy = self.energy(self.new_config)


    def u_v_v(self, q):
        theta = q[:,:,0]
        phi = q[:, :, 1]
        n = np.zeros((self.N, self.N, 3 ))
        n[...,0] += np.sin(theta) * np.cos(phi)
        n[...,1] += np.sin(theta) * np.sin(phi)
        n[...,2] += np.cos(theta)
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


    def observables(self, **params):
        if params["hmc"]:
            self.energy_list.append(self.energy_value)
            self.magnetisation_list.append(self.magnetisation())
        else:
            if self.iteration % self.N ** 2 == 0:
                self.energy_list.append(self.energy_value)
                self.magnetisation_list.append(self.magnetisation())


    def stop(self):
        return self.energy_list

    def leap_frog(self, q, p):
        for i in range(self.leap_steps):
            p = p - self.dHdq(q, self.nb(q)) * self.epsilon / 2
            q = q + p * self.epsilon
            dHdq= self.dHdq(q, self.nb(q))
            p = p - dHdq * self.epsilon / 2
        return q,p
