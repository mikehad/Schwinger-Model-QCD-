# Schwinger-Model-QCD-
-Measurments, is the Parent-class. -Schwinger contains the forces and energ functions. -Integraotrs contains various integrators. -Observable class is for data analysis. mc2py is the package

One can start with the following to create an object:

env=mc2py.Schwinger(N=16, beta=2, solver=lambda q, p: mc2py.Integrators.leap_frog(env, q, p),mass = 0.1, i=12, mu= 0.0,tau=1)
