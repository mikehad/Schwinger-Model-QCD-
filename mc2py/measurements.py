"""
env
"""

from time import time
from functools import wraps
import numpy as np
import copy


class Environment:
    def __init__(self, **params):
        for key, val in params.items():
            setattr(self, key, val)

    def initial_config(self, N):  # initialises the configuration
        raise NotImplementedError()

    def move_on(self, **params):
        if "max_iterations" in params:
            return self.iteration < params["max_iterations"]
        if "duration" in params:
            return self.start_time + params["duration"] > time()
        raise NotImplementedError("No param given for deciding on move_on")

    def update(self, **params):
        if params["hmc"]:
            self.generation_parameters()
            self.energy_value = self.energy(self.config) + 0.5 * np.sum(
                self.global_momentum ** 2
            )
            phase_space = self.solver(self.config.copy(), self.global_momentum.copy())
            self.new_config = phase_space[0]
            self.new_energy = self.energy(phase_space[0]) + 0.5 * np.sum(
                phase_space[1] ** 2
            )
        else:
            self.change_one_site()
        self.iteration += 1

    def update_energy(self, params):
        raise NotImplementedError()

    def accept(self, **params):
        delta_E = self.new_energy - self.energy_value
        self.dE_list.append(delta_E)
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E):
            return True
        return False

    def energy(self, **params):
        raise NotImplementedError()

    def magnetisation(self, **params):
        raise NotImplementedError()

    def finalize(self, **params):
        self.config = self.new_config.copy()
        self.energy_value = self.new_energy.copy()
        self.accepting += 1

    def observables(self, **params):
        raise NotImplementedError()

    def dSdq(self):
        raise NotImplementedError()

    def plaquette(self):
        raise NotImplementedError()

    def get_params(self, **params):
        if "duration" in params and "max_iterations" in params:
            raise ValueError("Only one between ..")
        if "duration" in params:
            self.start_time = time()
        else:
            self.iteration = 0
            params.setdefault("max_iterations", 100)
        params.setdefault("hmc", False)

        return params

    def change_one_site(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()
