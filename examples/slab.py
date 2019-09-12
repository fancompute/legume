import matplotlib.pyplot as plt
import numpy as np

from pygme.guided_modes_new import guided_modes

g_array = np.linspace(0.1, 13, 30)
eps_array = np.array([1.0, 12, 1.0, 12, 1.0])
d_array = np.array([1.0, 0.1, 1.0])

(omegas_TE, _) = guided_modes(g_array, eps_array, d_array, n_modes=8, step=1e-3, tol=1e-4, mode='TE')
(omegas_TM, _) = guided_modes(g_array, eps_array, d_array, n_modes=8, step=1e-3, tol=1e-4, mode='TM')

plt.figure()
for i, (omega_TE, omega_TM) in enumerate(zip(omegas_TE, omegas_TM)):
    plt.plot(g_array[i], np.atleast_2d(omega_TE), 'ok')
    plt.plot(g_array[i], np.atleast_2d(omega_TM), 'or')

plt.show()
plt.xlabel("Wave vector")
plt.ylabel("Frequency")
