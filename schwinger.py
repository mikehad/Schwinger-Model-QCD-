from functools import wraps
import numpy as np
from .measurements import Environment
from .integrators import Integrators
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator


class Schwinger(Environment):
    def __init__(self, N, beta, solver, mass, mu, i, tau):
        self.N = N
        self.beta = beta
        self.accepting = 0
        self.config = self.initial_config()
        self.new_config = np.zeros((self.N, self.N, 2))
        self.global_momentum = np.zeros(self.config.shape)
        self.energy_value = 0.0
        self.new_energy = 0.0
        self.m = mass
        self.energy_list = []
        self.dE_list = []
        self.F_SF_list = []
        self.F_SG_list = []
        self.trajectory_steps = i
        self.M = 3
        self.tau = tau
        self.mu = mu
        self.epsilon = self.tau / self.trajectory_steps
        self.eta = np.zeros(self.config.shape, dtype=complex)
        self.ksi = np.zeros(self.config.shape, dtype=complex)
        self.chi = np.zeros(self.config.shape, dtype=complex)
        self.F_SF_ = np.zeros(self.config.shape, dtype=complex)
        self.F_SG_ = np.zeros(self.config.shape, dtype=complex)
        self.solver = solver

    def initial_config(self):
        return np.exp(2j * np.pi * np.random.rand(self.N, self.N, 2))

    def plaquette(self, config):
        plaquette = (
            config[..., 0]
            * np.roll(config, -1, axis=0)[..., 1]
            * np.conjugate(np.roll(config, -1, axis=1)[..., 0])
            * np.conjugate(config[..., 1])
        )
        return plaquette

    def mul(self, a, b):
        return np.einsum("ij,ijk -> ijk", a, b)

    def energy(self, config):
        return self.gauge_action(config) + self.fermionic_action()

    def gauge_action(self, config):
        return self.beta * np.sum(1 - np.real(self.plaquette(config)))

    def fermionic_action(self):
        return np.sum(self.chi.conj() * self.chi)

    def mv_sigma3_D(self, eta, config):
        eta = eta.reshape(self.eta.shape)
        return self.D(self.sigma3(eta), config).flatten()

    def mv_D(self, eta, config):
        eta = eta.reshape(self.eta.shape)
        return self.D(eta, config).flatten()

    def mv_D_dagger(self, eta, config):
        eta = eta.reshape(self.eta.shape)
        return self.D_dagger(eta, config).flatten()

    def generation_parameters(self):
        self.global_momentum = np.random.normal(0.0, 1.0, self.config.shape)
        self.chi = np.random.normal(0.0, 1.0, self.config.shape)
        self.eta = self.D_dagger(self.chi, self.config)

    def eigensolver(self, config, k):
        eigenvalues = eigs(LinearOperator(
            (self.N * self.N * 2, self.N * self.N * 2),
            matvec=lambda x: self.mv_D(x, config),
            dtype=complex
        ), k = k
    )[0]
        return eigenvalues

    def dSdq(self, q):
        self.F_SG_ = self.F_SG(q)
        self.F_SF_ = self.F_SF(q)
        return self.F_SG_ + self.F_SF_

    def F_SG(self, q):
        plaquette = self.plaquette(q)
        F_SG = np.zeros(q.shape)
        F_SG[..., 0] = self.beta * np.imag(plaquette - np.roll(plaquette, 1, axis=1))
        F_SG[..., 1] = -self.beta * np.imag(plaquette - np.roll(plaquette, 1, axis=0))
        self.F_SG_list.append(np.sqrt(np.sum(F_SG**2)))
        return F_SG


    def F_SF(self, config):
        F_SF = np.zeros(self.config.shape, dtype=complex)
        self.chi = bicgstab(
            LinearOperator(
                (self.N * self.N * 2, self.N * self.N * 2),
                matvec=lambda x: self.mv_D_dagger(x, config),
                dtype=complex,
            ),
            self.eta.flatten(),tol=10e-9
        )[0]
        self.chi = self.chi.reshape(self.eta.shape)
        self.ksi = bicgstab(
            LinearOperator(
                (self.N * self.N * 2, self.N * self.N * 2),
                matvec=lambda x: self.mv_D(x, config),
                dtype=complex,
            ),
            self.chi.flatten(),tol=10e-9
        )[0]
        self.ksi = self.ksi.reshape(self.eta.shape)
        F_SF[..., 0] = -np.imag(
            np.sum(
                self.chi.conj()
                * self.sigma0minus(
                    self.mul(config[..., 0], np.roll(self.ksi, -1, axis=0))
                ),
                axis=2,
            )
            - np.sum(
                np.roll(self.chi.conj(), -1, axis=0)
                * self.sigma0plus(self.mul(config[..., 0].conj(), self.ksi)),
                axis=2,
            )
        )
        F_SF[..., 1] = -np.imag(
            np.sum(
                self.chi.conj()
                * self.sigma1minus(
                    self.mul(config[..., 1], np.roll(self.ksi, -1, axis=1))
                ),
                axis=2,
            )
            - np.sum(
                np.roll(self.chi.conj(), -1, axis=1)
                * self.sigma1plus(self.mul(config[..., 1].conj(), self.ksi)),
                axis=2,
            )
        )
        self.F_SF_list.append(np.sqrt(np.sum(F_SF**2)))
        return F_SF

    def sigma0plus(self, x):
        return x + np.roll(x, 1, axis=2)

    def sigma1plus(self, x):
        return x + np.roll(x, 1, axis=2) * np.array([-1j, 1j])

    def sigma0minus(self, x):
        return x - np.roll(x, 1, axis=2)

    def sigma1minus(self, x):
        return x - np.roll(x, 1, axis=2) * np.array([-1j, 1j])

    def sigma0(self, x):
        return np.roll(x, 1, axis=2)

    def sigma1(self, x):
        return np.roll(x, 1, axis=2) * np.array([-1j, 1j])

    def sigma3(self, x):
        return x * np.array([1, -1])

    def D(self, x, config):
        y = (2 + self.m) * x - 0.5 * (
            self.sigma0minus(self.mul(config[..., 0], np.roll(x, -1, axis=0)))
            + self.sigma0plus(
                self.mul(
                    np.roll(config[..., 0].conj(), 1, axis=0), np.roll(x, 1, axis=0)
                )
            )
            + self.sigma1minus(self.mul(config[..., 1], np.roll(x, -1, axis=1)))
            + self.sigma1plus(
                self.mul(
                    np.roll(config[..., 1].conj(), 1, axis=1), np.roll(x, 1, axis=1)
                )
            )
        )+ 1j*self.sigma3(x)*self.mu
        return y

    def D_dagger(self, x, config):
        y = (2 + self.m) * x - 0.5 * (
            self.sigma0plus(self.mul(config[..., 0], np.roll(x, -1, axis=0)))
            + self.sigma0minus(
                self.mul(
                    np.roll(config[..., 0].conj(), 1, axis=0), np.roll(x, 1, axis=0)
                )
            )
            + self.sigma1plus(self.mul(config[..., 1], np.roll(x, -1, axis=1)))
            + self.sigma1minus(
                self.mul(
                    np.roll(config[..., 1].conj(), 1, axis=1), np.roll(x, 1, axis=1)
                )
            )
        )-1j*self.sigma3(x)*self.mu
        return y

    def C_GG(self, config):
        P1 = np.zeros(config.shape, dtype=complex)
        P2 = np.zeros(config.shape, dtype=complex)
        P3 = np.zeros(config.shape, dtype=complex)
        P4 = np.zeros(config.shape, dtype=complex)
        P5 = np.zeros(config.shape, dtype=complex)
        P6 = np.zeros(config.shape, dtype=complex)
        P7 = np.zeros(config.shape, dtype=complex)
        P8 = np.zeros(config.shape, dtype=complex)
        p = self.plaquette(config)
        P1[..., 0] = p
        P1[..., 1] = P1[..., 0].conj()
        P2[..., 0] = np.roll(p, 1, axis=1)
        P2[..., 1] = np.roll(p.conj(), 1, axis=0)
        P3[..., 0] = np.roll(p, -1, axis=0)
        P3[..., 1] = np.roll(p.conj(), -1, axis=1)
        P4[..., 0] = P3[..., 1].conj()
        P4[..., 1] = P3[..., 0].conj()
        P5[..., 0] = P2[..., 1].conj()
        P5[..., 1] = P2[..., 0].conj()
        P6[..., 0] = np.roll(np.roll(p, 1, axis=0), 1, axis=1)
        P6[..., 1] = P6[..., 0].conj()
        P7[..., 0] = np.roll(p, 2, axis=1)
        P7[..., 1] = np.roll(p.conj(), 2, axis=0)
        P8[..., 0] = np.roll(np.roll(p, 1, axis=1), -1, axis=0)
        P8[..., 1] = np.roll(np.roll(p.conj(), 1, axis=0), -1, axis=1)
        C_GG = (
            2
            * (self.beta ** 2)
            * (
                np.imag(4 * P1 - P2 - P3 - P4 - P5) * np.real(P1)
                - np.imag(4 * P2 - P1 - P6 - P7 - P8) * np.real(P2)
            )
        )
        return C_GG

    def update_energy(self):
        cost = self.small_plaquette(self.a, self.b)
        if self.c == 0:
            cost += self.small_plaquette((self.a), (self.b - 1) % self.N)
        else:
            cost += self.small_plaquette((self.a - 1) % self.N, (self.b) % self.N)
        return self.beta * (1 - np.real(cost))

    def change_one_site(self):
        self.a = np.random.randint(0, self.N)
        self.b = np.random.randint(0, self.N)
        self.c = np.random.randint(0, 2)
        self.new_config = self.config.copy()
        self.new_config[self.a, self.b, self.c] = np.exp(2j * np.random.rand() * np.pi)
        self.new_energy = self.energy(self.new_config)

    def stop(self, **params):
        return self.energy_list, self.dE_list, self.accepting / params["max_iterations"], self.F_SF_list, self.F_SG_list

    def observables(self, **params):
        if params["hmc"]:
            self.energy_list.append(self.energy_value)
        else:
            if self.iteration % self.N ** 2 == 0:
                self.energy_list.append(self.energy_value)


def metropolis(env, **params):  # metropolis
    params = env.get_params(**params)
    while env.move_on(**params):
        env.update(**params)
        if env.accept(**params):
            env.finalize(**params)
        env.observables(**params)
    print(env.accepting)
    return env.stop(**params)
