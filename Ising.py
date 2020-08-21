class Ising(Environment):
    def __init__(self, N, T):
        self.energy_list = []
        self.magnetisation_list = []
        self.iteration = 0
        self.accepting = 0
        self.a = 0
        self.b = 0
        self.N = N
        self.T = T
        self.beta = 1 / T
        self.config = self.initialstate()
        self.new_config = np.zeros((self.N, self.N))
        self.energy_value = self.energy(self.config)
        self.new_energy = 0.0

    def initialstate(self):
        return 2 * np.random.randint(2, size=(self.N, self.N)) - 1

    def energy(self, config, **params):
        nb = (
            np.roll(config, 1, axis=0)
            + np.roll(config, 1, axis=1)
        )
        cost = np.sum(config[:, :] * nb[:, :])
        return -cost / 2

    def change_one_site(self, **params):
        self.a = np.random.randint(0, self.N)
        self.b = np.random.randint(0, self.N)
        self.new_config = self.config.copy()
        self.new_config[self.a, self.b] = self.new_config[self.a, self.b] * -1
        # use update_energy and not full energy
        self.new_energy = self.energy(self.new_config)

    def update_energy(self, **params):
        a = self.a
        b = self.b
        N = self.N
        nb = (
            self.config[(a + 1) % N, b]
            + self.config[a, (b + 1) % N]
            + self.config[(a - 1) % N, b]
            + self.config[a, (b - 1) % N]
        )
        cost = 2 * self.config[a, b] * nb
        return cost

    def stop(self):
        return self.energy_list

    def observables(self, **params):
        if params["hmc"]:
            self.energy_list.append(self.energy_value)
            self.magnetisation_list.append(self.magnetisation())
        else:
            if self.iteration % self.N ** 2 == 0:
                self.energy_list.append(self.energy_value)
                self.magnetisation_list.append(self.magnetisation())
