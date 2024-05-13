import numpy as np
import matplotlib.pyplot as plt
from mapping import StandardDomainMapping, StandardTimeMapping
from plotter import Plotter3D
from reconstruction import Reconstruction
from solver import Solver, MESolver
from analyzer import Analyzer


if __name__ == "__main__":
    np.printoptions(linewidth=np.inf)
    N0 = 10

    d_map = StandardDomainMapping("crazy_mesh", c=0.2)
    t_map = StandardTimeMapping("linear", t_begin=0, t_end=2.)

    s = Solver(problem_id=1,
               sparse=False, verbose=False,
               N=N0, N_int_x=N0, N_int_y=N0, N_int_t=N0, d_map=d_map, t_map=t_map)

    s.print_problem()

    mes = MESolver(problem_id=1,
                   K = 1,
                   sparse=False, verbose=False,
                   N=N0, N_int_x=N0, N_int_y=N0, N_int_t=N0, d_map=d_map, t_map=t_map)
    
    mes.print_problem()
    
    N = np.arange(2, 8, 1, dtype=int)
    K = np.arange(2, 8, 1, dtype=int)
    N_int = (max(N) + 1) * np.ones(len(N), dtype=int)

    analyzer = Analyzer(mes)
    error, _ = analyzer.compute_L2_zw_h(dual=False,
                                        N=np.ones_like(N) * N0,
                                        K=K,
                                        N_int_x=N_int, N_int_y=N_int, N_int_t=N_int,
                                        N_L2_int_x=N_int, N_L2_int_y=N_int, N_L2_int_t=N_int,
                                        regression=True
                                        )
    # Plot the error on the log scale
    plt.plot(K, error, 'bo-')
    plt.yscale('log')
    plt.show()
    
    # analyzer.plot_L2_zw_h(False, N, zw_exact, N_int_x=N_int, N_int_y=N_int, N_int_t=N_int, mode="def", regression=True)
    
    # e_tot = analyzer.compute_energy([0., 0.5, 1, 1.5, 2.], 15, 15)
    # analyzer.plot_energy(np.linspace(0., 2., 100), N_int_x=N0+0, N_int_y=N0+0)
    

