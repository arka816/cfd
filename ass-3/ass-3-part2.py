import numpy as np
import matplotlib.pyplot as plt

radius = 0.05
alpha = 1.16e-5
T_init = 400
k = 45

T_inf_upper = 20
h_upper = 50

T_inf_lower = 5
h_lower = 250 

del_r = 0.0025
del_t = 0.0025

C = alpha * del_t / (del_r ** 2)

N_r = int(radius / del_r + 1)
N_theta = 8
del_theta = 2 * np.pi / N_theta

T = np.full((N_r, N_theta), T_init)
time = 0
iters = 0

theta = np.linspace(0.0, 2 * np.pi, N_theta + 1)
r = np.linspace(0, radius, N_r)
grid_r, grid_theta = np.meshgrid(r, theta)

while np.max(T) > 200:
    T_new = np.zeros((N_r, N_theta))
    for j in range(N_theta):
        # center i.e. r = 0 i.e. i = 1
        i = 0
        T_new[i, j] = T[i, j] * (1 - 4 * C + 2 * C / (del_theta ** 2))
        + T[i+1, j] * (4 * C - 2 * C / (del_theta ** 2))
        + (T[i+1, (j+1)%N_theta] - T[i, (j+1)%N_theta] + T[i+1, (j-1)%N_theta] - T[i, (j-1)%N_theta]) * C / (del_theta ** 2)
        
        # alternate
        # T_new[i, j] = T[i, j] * (1 - 4 * C) + (T[i+1, j] + T[i-1, j] + T[i, (j+1)%N_theta] + T[i, (j-1)%N_theta]) * C
        
        # all internal points
        for i in range(1, N_r - 1):
            T_new[i, j] = T[i, j] * (1 - 2 * C * (1 + 1 / ((i * del_theta) ** 2)))
            + T[i+1, j] * C * (1 + 1/(2 * i)) + T[i-1, j] * C * (1 - 1/(2 * i))
            + (T[i, (j+1)%N_theta] + T[i, (j-1)%N_theta]) * C / ((i * del_theta) ** 2)
            
        # surface
        i = N_r - 1
        if j < N_theta // 2:
            # for upper surface
            T_new[i, j] = T[i, j] * (1 - 2 * C - 2 * C * h_upper * del_r * (1 + 1 / (2 * i)) / k - 2 * C / ((i * del_theta) ** 2))
            + 2 * C * T[i-1, j] + (T[i, (j+1)%N_theta] + T[i, (j-1)%N_theta]) * C / ((i * del_theta) ** 2) 
            + 2 * C * h_upper * del_r * T_inf_upper * (1 + 1 / (2 * i)) / k
        else:
            # for lower surface
            T_new[i, j] = T[i, j] * (1 - 2 * C - 2 * C * h_lower * del_r * (1 + 1 / (2 * i)) / k - 2 * C / ((i * del_theta) ** 2))
            + 2 * C * T[i-1, j] + (T[i, (j+1)%N_theta] + T[i, (j-1)%N_theta]) * C / ((i * del_theta) ** 2) 
            + 2 * C * h_lower * del_r * T_inf_lower * (1 + 1 / (2 * i)) / k
            
    T = T_new
    time += del_t
    iters += 1

    # plt.figure()
    # ax1 = plt.subplot(projection="polar")
    # cm = ax1.pcolormesh(theta,r,T, shading='auto', cmap='inferno', vmin=np.min(T), vmax=np.max(T))
    # plt.colorbar(cm)
    # plt.show()
