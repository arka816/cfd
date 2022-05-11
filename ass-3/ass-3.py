import numpy as np
import matplotlib.pyplot as plt

radius = 0.05
alpha = 1.16e-5
T_init = 400
T_inf = 20
h = 50
k = 45

delr = 0.0025
delt = 0.125

N_r = int(radius / delr + 1)
H = alpha * delt / (delr ** 2)
P = h * delr / k

T = np.full(N_r, T_init)

time = 0

while T[0] > 200:
    T_new = np.empty(N_r)
    
    # Boundary condition at R = 0
    T_new[0] = (1 - 4*H)*T[0] + 4*H*T[1]
    
    for i in range(1, N_r-1):
        T_new[i] = (1 - 2*H)*T[i] + H*(1 + 1 / (2*i))*T[i+1] + H*(1 - 1/(2*i))*T[i-1]
    
    # Boundary Condition at R = 1
    T_new[-1] = (1 - 2*H - 2*P*H*(1 + 1/(2 * (N_r - 1)))) * T[-1] + 2*H*T[-2] + 2*P*H*(1 + 1/(2 * (N_r - 1)))*T_inf
    T = T_new
    
    time += delt
    
print(time/60, "minutes")

plt.plot(np.arange(0, radius + delr, delr), T)
plt.title("radial temperature distribution")
plt.ylabel("temperature (in °C)")
plt.xlabel("radial distance (in metre)")
plt.show()
