# Partial translation of fem_timing.jl to Python
import time
from fem import stiffness_matrix, define_load, fix_support

if __name__ == "__main__":
    dx = 0.1
    nelx = 255
    nely = 255

    print("Assemble stiffness matrix, assemble right hand side, and set boundary conditions")
    t0 = time.time(); K = stiffness_matrix(nely, nelx); print("stiffness assembled", time.time()-t0)
    t0 = time.time(); F = define_load(nely, nelx, dx, lambda x, y: y); print("load assembled", time.time()-t0)
    t0 = time.time(); K, F = fix_support(nely, nelx, K, F); print("fixed", time.time()-t0)

    print("Solving K U = F using numpy.linalg.solve ...")
    t0 = time.time();
    import numpy as np
    U = np.linalg.solve(K.toarray(), F)
    print("solve time:", time.time()-t0)
