import numpy as np
from gssor import GSSOR
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from scipy import ndimage

PI = np.pi
LEVEL_THRESHOLD = 0.02
mesh_sizes = [10, 20, 30, 35, 40]


def genSystem_GS1(mesh_size):
    N = mesh_size
    unit = 1 / (N-1)
    
    A = np.zeros((N**2, N**2))
    b = np.zeros(N**2)
    
    for i in range(N**2):
        if i < N:
            A[i, i] = 1
            b[i] = 100 * np.sin(PI * unit * (i))
        elif i < N * (N-1):
            A[i, i] = -4
            A[i, i-N] = 1
            A[i, i+N] = 1
            if i%N == 0:
                A[i, i+1] = 2
            elif i%N == N-1:
                A[i, i-1] = 2
            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
        else:
            A[i, i] = -4
            A[i, i-N] = 2
            if i%N == 0:
                A[i, i+1] = 2
            elif i%N == N-1:
                A[i, i-1] = 2
            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
                
    return A, b

def genSystem_GS2(mesh_size):
    N = mesh_size
    unit = 1 / (N-1)
    
    A = np.zeros((N**2, N**2))
    b = np.zeros(N**2)
    
    T_inf = 30
    h = 20
    k = 30
    const = 2 * h * unit / k
    
    for i in range(N**2):
        if i < N:
            A[i, i] = 1
            b[i] = 100 * np.sin(PI * unit * (i))
        elif i < N * (N-1):
            A[i, i] = -4
            A[i, i-N] = 1
            A[i, i+N] = 1
            if i%N == 0:
                A[i, i+1] = 2
            elif i%N == N-1:
                A[i, i-1] = 2
                A[i, i] = -4 - const
                b[i] = -const * T_inf
            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
        else:
            A[i, i] = -4
            A[i, i-N] = 2
            
            if i%N == 0:
                A[i, i+1] = 2
            elif i%N == N-1:
                A[i, i-1] = 2
                A[i, i] = -4 - const
                b[i] = -const * T_inf
            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
                
    return A, b

def selectLevels(x):
    unique, counts = np.unique(x.astype(np.int32), return_counts=True)
    total_count = x.shape[0] * x.shape[1]
    
    vals = []
    
    for val, count in zip(unique, counts):
        if count / total_count >= LEVEL_THRESHOLD:
            vals.append(val)
            
    return tuple(sorted([min(unique) - 1] + [i - 0.5 for i in vals] + [vals[-1] + 0.5]))
    

def solve(case):
    diagonals = []
    
    for mesh_size in mesh_sizes:
        # 1. SOLVE USING gauss-seidel using successive over-relaxation
        # 2. also determine best rf for each mesh
        # 3. also demonstrate grid independence using diagonal values
        
        if case == 1:
            A, b = genSystem_GS1(mesh_size)
        elif case == 2:
            A, b = genSystem_GS2(mesh_size)
        x, best_sor, best_iters = GSSOR(A, b)
        
        print(f'best SOR rf for mesh size ({mesh_size}x{mesh_size}) :', best_sor)
        print('best iterations obtained:', best_iters)
        
        x = np.reshape(x, (mesh_size, mesh_size))
        diagonals.append((mesh_size, x.diagonal()))
        x = np.flip(x, axis=0)
        
        # plot heatmap and contour lines
        fig, ax = plt.subplots()
        smooth_scale = 5    
        z = ndimage.zoom(x, smooth_scale)
        cntr = ax.contourf(
            np.linspace(0, mesh_size, mesh_size * smooth_scale),
            np.linspace(0, mesh_size, mesh_size * smooth_scale),
            z, 
            levels=20, 
            cmap='inferno'
        )
        ax = sns.heatmap(x, annot=False, alpha=0, cbar=False, ax = ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.title('2-d temperature distribution')
        plt.colorbar(cntr, ax=ax)
        # plt.contour(x, levels)
        # plt.contourf(x, levels = np.unique(x))
        # sns.heatmap(x, annot=True, cbar=True)
        # plt.show()
        # c = pl.contourf(x)
        # pl.colorbar(c)
        del A, b, x
        
    x = np.arange(0, 30, 1)
    plt.figure()
    plt.title('grid independence demonstration')
    for mesh_size, diagonal in diagonals:
        y_interp = np.interp(x, np.arange(0.5, mesh_size, 1) * 30 / mesh_size, diagonal)
        label = f'{mesh_size} x {mesh_size}'
        plt.plot(x, y_interp, label=label, marker='o')
    plt.legend()
    plt.show()

solve(2)
