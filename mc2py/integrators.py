from functools import wraps
import numpy as np
from .measurements import Environment
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import LinearOperator

class Integrators:

    def leap_frog(self, q, p):
        p = p - self.epsilon *0.5 * self.dSdq(q)
        for it in range(self.trajectory_steps):
            q = q * np.exp(complex(0, 1) * self.epsilon * p)
            f = self.dSdq(q)
            if it != self.trajectory_steps - 1:
                p = p - self.epsilon * f
        p = p - self.epsilon*0.5 * f
        return q, p

    def leap_frog_stupid(self, q, p):
        for i in range(self.trajectory_steps):
            p = p - 0.5 * self.F_SF(q) * self.epsilon
            for j in range(self.trajectory_steps2):
                p = p - 0.5 * self.F_SG(q) * self.epsilon/self.trajectory_steps2
                q = q * np.exp(1j * p * self.epsilon/self.trajectory_steps2)
                p = p - 0.5 * self.F_SG(q) * self.epsilon/self.trajectory_steps2
            p = p - 0.5 * self.F_SF(q) * self.epsilon
        return q, p


    def five_stage_nested(self, q, p):
        for i in range(self.trajectory_steps):
            p = p - self.dSdq(q) * self.epsilon / 6
            self.leap_frog_gauge()
            p = p - self.dSdq(q) * 2 * self.epsilon / 3
            self.leap_frog_gauge()
            p = p - self.dSdq(q) * self.epsilon / 6
        return q, p

    def five_stage(self, q, p):
        for i in range(self.trajectory_steps):
            p = p - self.dSdq(q) * self.epsilon / 6
            q = q * np.exp(p * 1j * self.epsilon / 2)
            p = p - self.dSdq(q) * 2 * self.epsilon / 3
            q = q * np.exp(p * 1j * self.epsilon / 2)
            p = p - self.dSdq(q) * self.epsilon / 6
        return q, p


    def fgs(self, q, p):
        for i in range(self.trajectory_steps):
            p = p - self.dSdq(q) * self.epsilon / 6
            q = q * np.exp(p * 1j * self.epsilon / 2)
            p = (
                p
                - self.dSdq(q) * 2 * self.epsilon / 3
                + self.C_GG(q) * (self.epsilon ** 3) / 72
            )
            q = q * np.exp(p * 1j * self.epsilon / 2)
            p = p - self.dSdq(q) * self.epsilon / 6
        return q, p

    def fgs_numerical(self, q, p):
        for i in range(self.trajectory_steps):
            p = p - self.dSdq(q) * self.epsilon / 6
            q = q * np.exp(p * 1j * self.epsilon / 2)
            force_old=self.dSdq(q)
            q_temp = q * np.exp(-complex(0, 1) * self.epsilon**2 * force_old/24)
            p = p -  self.dSdq(q_temp)*2*self.epsilon/3
            q = q * np.exp(p * 1j * self.epsilon / 2)
            p = p - self.dSdq(q) * self.epsilon / 6
        return q, p

    def fgs_pure_numerical(self, q, p):
        for i in range(self.trajectory_steps):
            p = p - self.F_SG(q) * self.epsilon / 6
            q = q * np.exp(p * 1j * self.epsilon / 2)
            force_old=self.F_SG(q)
            q_temp = q * np.exp(-complex(0, 1) * self.epsilon**2 * force_old/24)
            p = p -  self.F_SG(q_temp)*2*self.epsilon/3
            q = q * np.exp(p * 1j * self.epsilon / 2)
            p = p - self.F_SG(q) * self.epsilon / 6
        return q, p

    def rk2(self, q, p):
        for i in range(self.trajectory_steps):
            q_old = q; p_old = p;
            p = p_old - 0.5*self.epsilon*self.dSdq(q_old)
            q = q_old *np.exp(1j * self.epsilon * 0.5* p_old)
            p = p_old - self.epsilon*self.dSdq(q)
            q = q_old*np.exp(1j * self.epsilon * p)
        return q, p


    def eleven_stage(
        self,
        q,
        p,
        rho=0.2539785108410595,
        theta=-0.03230286765269967,
        sigma=0.08398315262876693,
        lambda_=0.6822365335719091,
    ):
        for i in range(self.trajectory_steps):
            p = p - self.dSdq(q) * self.epsilon * sigma
            q = q * np.exp(p * 1j * self.epsilon * rho)
            p = p - self.dSdq(q) * self.epsilon * lambda_
            q = q * np.exp(p * 1j * self.epsilon * theta)
            p = p - self.dSdq(q) * self.epsilon * 0.5 * (1 - 2 * lambda_ - 2 * sigma)
            q = q * np.exp(p * 1j * self.epsilon * (1 - 2 * (theta + rho)))
            p = p - self.dSdq(q) * self.epsilon * 0.5 * (1 - 2 * lambda_ - 2 * sigma)
            q = q * np.exp(p * 1j * self.epsilon * theta)
            p = p - self.dSdq(q) * self.epsilon * lambda_
            q = q * np.exp(p * 1j * self.epsilon * rho)
            p = p - self.dSdq(q) * self.epsilon * sigma
        return q, p
