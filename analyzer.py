import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from assignment_functions import lobatto_quad
from mapping import StandardDomainMapping, StandardTimeMapping
from components import Metric
from solver import Problem, Solver, MESolver
from typing import Union, Callable, Literal
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib import rc


class PlotterHelper(object):

    MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'h', 'H', '+', 'x', '|', '_']
    COLORS = ["#CC6677", "#999933", "#117733", "#332288"][::-1]
    MARKERSIZE = 3
    REC_RATIO = 4 / 4.   # w / h
    A4_DIMS = (8.3, 11.7)

    @staticmethod
    def adjust_font():
        plt.rcParams.update({
                "text.usetex": True,  # Use TeX for rendering
                "font.family": "serif",  # Use serif font
                "font.serif": ["cmr10"],  # Set the CMR font family and size
                "font.size": 10,  # Set the default font size
                "axes.formatter.use_mathtext": True,  # Enable mathtext mode
            })
            
        rc('text.latex', preamble=r'\usepackage{color}')

    @staticmethod
    def get_figsize(n_figs_on_page:int = 2, ratio: float=REC_RATIO, page_margin: float=1):
        w = (PlotterHelper.A4_DIMS[0] - 2 * page_margin) / n_figs_on_page
        h = w / ratio

        return (w, h)

    @staticmethod
    def get_colors_markers_markersize():
        return PlotterHelper.COLORS, PlotterHelper.MARKERS, PlotterHelper.MARKERSIZE

class Analyzer(object):

    def __init__(self, top_solver: Union[Solver, MESolver]) -> None:
        
        # Check typing
        if not isinstance(top_solver, Solver) and not isinstance(top_solver, MESolver):
            return TypeError(f"top_solver must be either of type Solver or MESolver, not {type(top_solver)}.")

        self.type: Literal['SE', 'ME'] = f"{ 'SE' if isinstance(top_solver, Solver) else 'ME' }"

        self.top_solver: Union[Solver, MESolver] = top_solver

        if self.type == 'SE':
            self.top_problem: Problem = top_solver.problem
        else:
            self.top_problem: Problem = top_solver.initial_problem

    @staticmethod
    def same_lengths(tab: list) -> bool:

        """
        Function checks if all variables passed in the tab argument have the same length.

        The function takes a list of variables. By assumption this list contains variables that can be either
        an int, np.ndarray, or list type. Function checks if all the variables have the same length 
        (an int has length 1 by definition). Raises RuntimeWarning if the passed list is empty. 
        Returns True if all variable lengths are equal, False otherwise.

        Parameters:
        ----------
        tab : list
            List containing variables to check for length equality.

        Returns:
        -------
        bool
            True if all variable lengths are equal, False otherwise.

        Raises:
        -------
        RuntimeWarning
            If the list of tabs is empty.
        """

        if len(tab) < 1:
            raise RuntimeWarning("The list of tabs is empty.")

        # initialize list of lengths
        lengths = []

        for i in range(len(tab)):
            lst: Union[int, np.ndarray, list] = tab[i]

            if isinstance(lst, int):
                lengths.append(1)
            else:
                lengths.append(len(lst))

        lengths = np.array(lengths)

        return np.all(lengths == lengths[0])

    # Private method for spatial pointwise error computation
    def __compute_abs_spat_ptwise_error_zw_h(self, dual: bool, \
                        t: float, \
                        N: int, K: int, \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        N_res_x: int, N_res_y: int) -> np.ndarray:
        
        # Assure that N is alway given
        if N is None:
            raise ValueError("N parameter must be provided.")

        # Assure that the exact solution exists in the problem (not None)
        if self.top_problem.zw_exact is None:
            raise ValueError("The exact solution of the problem zw_exact must be provided (not None).")

        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        
        # If multiple element solver is used, the K parameter must be provided
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided by keyword K when using multi element solver.")        

        # Create auxiliary solver: the same args as the top_solver but update orders
        aux_kwargs = self.top_solver.kwargs
        aux_kwargs["N"] = N

        # Order of mass matrix integration
        aux_kwargs["N_int_x"] = N + DN_int_x
        aux_kwargs["N_int_y"] = N + DN_int_y
        aux_kwargs["N_int_t"] = N + DN_int_t

        if self.type == 'SE':
            aux_solver = Solver(self.top_solver.problem_id, self.top_solver.sparse, verbose=False, **aux_kwargs)
            # Solve with solver
            aux_solver.solve_system(verbose=False)
        else:
            aux_solver = MESolver(self.top_solver.problem_id, K, self.top_solver.sparse, verbose=False, **aux_kwargs)
            # Solve with solver
            aux_solver.solve(verbose=False)

        
        # Metric to be used in reconstruction
        metric = Metric(N, N+DN_int_x, N+DN_int_y, N+DN_int_t, self.top_problem.c, self.top_problem.r, self.top_problem.d_map, self.top_problem.t_map)

        # Retrieve mappings
        d_map = self.top_problem.d_map

        # Generate unit reconstruction grid
        xi_rec = np.linspace(-1, 1, N_res_x)
        eta_rec = np.linspace(-1, 1, N_res_y)
        mesh_xieta_rec = np.meshgrid(xi_rec, eta_rec)

        x_rec_grid = d_map.x(mesh_xieta_rec[0], mesh_xieta_rec[1])
        y_rec_grid = d_map.y(mesh_xieta_rec[0], mesh_xieta_rec[1])


        if dual:
            z_h = aux_solver.reconstruct_w(mesh_xieta_rec, t, verbose=False)   # solve for w DOFs
        else:
            z_h = aux_solver.reconstruct_z(mesh_xieta_rec, t, verbose=False)   # solve for z DOFs

        z_h = z_h.reshape(N_res_y, N_res_x)        

        # Evaluate the exact solution at integration nodes
        zw_ex_evaluated = self.top_problem.zw_exact(x_rec_grid, y_rec_grid, t)
        
        # Compute the absolute error
        return np.abs(zw_ex_evaluated - z_h)

    # Private method for spatial pointwise error computation
    def __compute_abs_spat_ptwise_error_pi_h(self, dual: bool, \
                        t: float, \
                        N: int, K: int, \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        N_res_x: int, N_res_y: int) -> np.ndarray:
        
        # Assure that N is alway given
        if N is None:
            raise ValueError("N parameter must be provided.")

        # Assure that the exact solution exists in the problem (not None)
        if self.top_problem.zw_exact is None:
            raise ValueError("The exact solution of the problem zw_exact must be provided (not None).")

        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        
        # If multiple element solver is used, the K parameter must be provided
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided by keyword K when using multi element solver.")        

        # Create auxiliary solver: the same args as the top_solver but update orders
        aux_kwargs = self.top_solver.kwargs
        aux_kwargs["N"] = N

        # Order of mass matrix integration
        aux_kwargs["N_int_x"] = N + DN_int_x
        aux_kwargs["N_int_y"] = N + DN_int_y
        aux_kwargs["N_int_t"] = N + DN_int_t

        if self.type == 'SE':
            aux_solver = Solver(self.top_solver.problem_id, self.top_solver.sparse, verbose=False, **aux_kwargs)
            # Solve with solver
            aux_solver.solve_system(verbose=False)
        else:
            aux_solver = MESolver(self.top_solver.problem_id, K, self.top_solver.sparse, verbose=False, **aux_kwargs)
            # Solve with solver
            aux_solver.solve(verbose=False)

        
        # Metric to be used in reconstruction
        metric = Metric(N, N+DN_int_x, N+DN_int_y, N+DN_int_t, self.top_problem.c, self.top_problem.r, self.top_problem.d_map, self.top_problem.t_map)

        # Retrieve mappings
        d_map = self.top_problem.d_map

        # Generate unit reconstruction grid
        xi_rec = np.linspace(-1, 1, N_res_x)
        eta_rec = np.linspace(-1, 1, N_res_y)
        mesh_xieta_rec = np.meshgrid(xi_rec, eta_rec)

        x_rec_grid = d_map.x(mesh_xieta_rec[0], mesh_xieta_rec[1])
        y_rec_grid = d_map.y(mesh_xieta_rec[0], mesh_xieta_rec[1])


        pi_h = aux_solver.reconstruct_pi(mesh_xieta_rec, t, verbose=False)   # solve for pi DOFs
        pi_h = pi_h.reshape(N_res_y, N_res_x)        

        # Evaluate the exact solution at integration nodes
        pi_ex_evaluated = self.top_problem.pi_exact(x_rec_grid, y_rec_grid, t)
        
        # Compute the absolute error
        return np.abs(pi_ex_evaluated - pi_h)
    
    # Private method for one-time evaluation of L2 error for zw
    def __compute_L2Linf_zw_h_def(self, \
                                dual: bool, \
                                N: int, \
                                K: Union[int, None]=None, \
                                N_int_x: Union[int, None]=None, N_int_y: Union[int, None]=None, N_int_t: Union[int, None]=None, \
                                N_L2_int_x: Union[int, None]=None, N_L2_int_y: Union[int, None]=None, N_L2_int_t: Union[int, None]=None, \
                                verbose: bool=False \
                                ) -> float:
        
        # Assure that N is alway given
        if N is None:
            raise ValueError("N parameter must be provided.")

        # Assure that the exact solution exists in the problem (not None)
        if self.top_problem.zw_exact is None:
            raise ValueError("The exact solution of the problem zw_exact must be provided (not None).")

        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        
        # If multiple element solver is used, the K parameter must be provided
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided by keyword K when using multi element solver.")

        # Check if any requirements were made regarding the orders
        if N_int_x is None:
            N_int_x = N

        if N_int_y is None:
            N_int_y = N

        if N_int_t is None:
            N_int_t = N

        if N_L2_int_x is None:
            N_L2_int_x = N
        
        if N_L2_int_y is None:
            N_L2_int_y = N
        
        if N_L2_int_t is None:
            N_L2_int_t = N
        

        # Create auxiliary solver: the same args as the top_solver but update orders
        aux_kwargs = self.top_solver.kwargs
        aux_kwargs["N"] = N


        # Order of mass matrix integration
        aux_kwargs["N_int_x"] = N_int_x
        aux_kwargs["N_int_y"] = N_int_y
        aux_kwargs["N_int_t"] = N_int_t

        if self.type == 'SE':
            aux_solver = Solver(self.top_solver.problem_id, self.top_solver.sparse, verbose=verbose, **aux_kwargs)
            # Solve with solver
            aux_solver.solve_system(verbose=False)
        else:
            aux_solver = MESolver(self.top_solver.problem_id, K, self.top_solver.sparse, verbose=verbose, **aux_kwargs)
            # Solve with solver
            aux_solver.solve(verbose=False)

        
        # Metric to be used in L2 INTEGRATION
        metric = Metric(N, N_L2_int_x, N_L2_int_y, N_L2_int_t, self.top_problem.c, self.top_problem.r, self.top_problem.d_map, self.top_problem.t_map)

        # Retrieve mappings
        d_map = metric.d_map
        t_map = metric.t_map

        # Generate unit integration grid
        xi_int_grid, tau_int_grid, eta_int_grid = np.meshgrid(metric.GL_int_x, metric.GL_int_y, metric.GL_int_t)
        mesh_xieta_int = np.meshgrid(metric.GL_int_x, metric.GL_int_y)

        x_int_grid = d_map.x(xi_int_grid, eta_int_grid)
        y_int_grid = d_map.y(xi_int_grid, eta_int_grid)
        t_int_grid = t_map.t(tau_int_grid)

        # Evaluate transformation metrics at unit integration nodes
        X_XI_INT = d_map.x_xi(xi_int_grid, eta_int_grid)
        X_ETA_INT = d_map.x_eta(xi_int_grid, eta_int_grid)
        Y_XI_INT = d_map.y_xi(xi_int_grid, eta_int_grid)
        Y_ETA_INT = d_map.y_eta(xi_int_grid, eta_int_grid)
        T_TAU_INT = t_map.t_tau(tau_int_grid)

        # The Jacobian evaluated at integration nodes
        J_INT = T_TAU_INT * (X_XI_INT * Y_ETA_INT - Y_XI_INT * X_ETA_INT)

        if dual:
            z_h = aux_solver.reconstruct_w(mesh_xieta_int, t_map.t(metric.GL_int_t), verbose=False)   # solve for w DOFs
        else:
            z_h = aux_solver.reconstruct_z(mesh_xieta_int, t_map.t(metric.GL_int_t), verbose=False)   # solve for z DOFs

        z_h = z_h.reshape(N_L2_int_y + 1, N_L2_int_x + 1, -1)        
        

        # Evaluate the exact solution at integration nodes
        zw_ex_evaluated = self.top_problem.zw_exact(x_int_grid, y_int_grid, t_int_grid)
        
        # Compute the difference squared with Jacobian
        d_sq = np.power(zw_ex_evaluated - z_h, 2)
        
        # Integrate
        I = np.einsum('ijk,i,j,k->', d_sq * J_INT, metric.weights_int_y, metric.weights_int_x, metric.weights_int_t, optimize='greedy')

        # Return the value sqrt, and Linf
        return np.sqrt(I), np.max(np.abs(zw_ex_evaluated - z_h))

    # Private method for one-time evaluation of L2 error for div
    def __compute_L2Linf_div_h_def(self, \
                                N: int, \
                                K: Union[int, None]=None, \
                                N_int_x: Union[int, None]=None, N_int_y: Union[int, None]=None, N_int_t: Union[int, None]=None, \
                                N_L2_int_x: Union[int, None]=None, N_L2_int_y: Union[int, None]=None, N_L2_int_t: Union[int, None]=None, \
                                verbose: bool=False \
                                ) -> float:
        
        # Assure that N is alway given
        if N is None:
            raise ValueError("N parameter must be provided.")

        # Assure that the exact solution exists in the problem (not None)
        if self.top_problem.zw_exact is None:
            raise ValueError("The exact solution of the problem zw_exact must be provided (not None).")

        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        
        # If multiple element solver is used, the K parameter must be provided
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided by keyword K when using multi element solver.")

        # Check if any requirements were made regarding the orders
        if N_int_x is None:
            N_int_x = N

        if N_int_y is None:
            N_int_y = N

        if N_int_t is None:
            N_int_t = N

        if N_L2_int_x is None:
            N_L2_int_x = N
        
        if N_L2_int_y is None:
            N_L2_int_y = N
        
        if N_L2_int_t is None:
            N_L2_int_t = N
        

        # Create auxiliary solver: the same args as the top_solver but update orders
        aux_kwargs = self.top_solver.kwargs
        aux_kwargs["N"] = N


        # Order of mass matrix integration
        aux_kwargs["N_int_x"] = N_int_x
        aux_kwargs["N_int_y"] = N_int_y
        aux_kwargs["N_int_t"] = N_int_t

        if self.type == 'SE':
            aux_solver = Solver(self.top_solver.problem_id, self.top_solver.sparse, verbose=verbose, **aux_kwargs)
            # Solve with solver
            aux_solver.solve_system(verbose=False)
        else:
            aux_solver = MESolver(self.top_solver.problem_id, K, self.top_solver.sparse, verbose=verbose, **aux_kwargs)
            # Solve with solver
            aux_solver.solve(verbose=False)

        
        # Metric to be used in L2 INTEGRATION
        metric = Metric(N, N_L2_int_x, N_L2_int_y, N_L2_int_t, self.top_problem.c, self.top_problem.r, self.top_problem.d_map, self.top_problem.t_map)

        # Retrieve mappings
        d_map = metric.d_map
        t_map = metric.t_map

        # Generate unit integration grid
        xi_int_grid, tau_int_grid, eta_int_grid = np.meshgrid(metric.GL_int_x, metric.GL_int_y, metric.GL_int_t)
        mesh_xieta_int = np.meshgrid(metric.GL_int_x, metric.GL_int_y)

        x_int_grid = d_map.x(xi_int_grid, eta_int_grid)
        y_int_grid = d_map.y(xi_int_grid, eta_int_grid)
        t_int_grid = t_map.t(tau_int_grid)

        # Evaluate transformation metrics at unit integration nodes
        X_XI_INT = d_map.x_xi(xi_int_grid, eta_int_grid)
        X_ETA_INT = d_map.x_eta(xi_int_grid, eta_int_grid)
        Y_XI_INT = d_map.y_xi(xi_int_grid, eta_int_grid)
        Y_ETA_INT = d_map.y_eta(xi_int_grid, eta_int_grid)
        T_TAU_INT = t_map.t_tau(tau_int_grid)

        # The Jacobian evaluated at integration nodes
        J_INT = T_TAU_INT * (X_XI_INT * Y_ETA_INT - Y_XI_INT * X_ETA_INT)

        div_h = aux_solver.reconstruct_div(mesh_xieta_int, t_map.t(metric.GL_int_t), verbose=False)   # solve for z DOFs

        div_h = div_h.reshape(N_L2_int_y + 1, N_L2_int_x + 1, -1)        
        
        # Compute the difference squared with Jacobian
        d_sq = np.power(div_h, 2)
        
        # Integrate
        I = np.einsum('ijk,i,j,k->', d_sq * J_INT, metric.weights_int_y, metric.weights_int_x, metric.weights_int_t, optimize='greedy')

        # Return the value sqrt and Linf
        return np.sqrt(I), np.max(np.abs(div_h))

    # Private method for one-time evaluation of condition number
    def __compute_condition_number(self, \
                                N: int, \
                                K: Union[int, None]=None, \
                                N_int_x: Union[int, None]=None, N_int_y: Union[int, None]=None, N_int_t: Union[int, None]=None, \
                                N_L2_int_x: Union[int, None]=None, N_L2_int_y: Union[int, None]=None, N_L2_int_t: Union[int, None]=None, \
                                mode: Literal['min', 'avg', 'max']='max', \
                                verbose: bool=False \
                                ) -> float:
        
        # Assure that N is alway given
        if N is None:
            raise ValueError("N parameter must be provided.")

        # Assure that the exact solution exists in the problem (not None)
        if self.top_problem.zw_exact is None:
            raise ValueError("The exact solution of the problem zw_exact must be provided (not None).")

        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        
        # If multiple element solver is used, the K parameter must be provided
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided by keyword K when using multi element solver.")

        # Check if any requirements were made regarding the orders
        if N_int_x is None:
            N_int_x = N

        if N_int_y is None:
            N_int_y = N

        if N_int_t is None:
            N_int_t = N

        if N_L2_int_x is None:
            N_L2_int_x = N
        
        if N_L2_int_y is None:
            N_L2_int_y = N
        
        if N_L2_int_t is None:
            N_L2_int_t = N
        

        # Create auxiliary solver: the same args as the top_solver but update orders
        aux_kwargs = self.top_solver.kwargs
        aux_kwargs["N"] = N


        # Order of mass matrix integration
        aux_kwargs["N_int_x"] = N_int_x
        aux_kwargs["N_int_y"] = N_int_y
        aux_kwargs["N_int_t"] = N_int_t

        if self.type == 'SE':
            aux_solver = Solver(self.top_solver.problem_id, self.top_solver.sparse, verbose=verbose, **aux_kwargs)
            # Solve with solver
            aux_solver.solve_system(verbose=False)
        else:
            aux_solver = MESolver(self.top_solver.problem_id, K, self.top_solver.sparse, verbose=verbose, **aux_kwargs)
            # Solve with solver
            aux_solver.solve(verbose=False)

        if mode == 'min':
            return np.min(aux_solver.get_condition_number())
        elif mode == 'avg':
            return np.mean(aux_solver.get_condition_number())
        elif mode == 'max':
            return np.max(aux_solver.get_condition_number())

    # Public method for L2 error for zw
    def compute_L2Linf_zw_h(self, \
                        dual: bool, \
                        N: Union[int, tuple, list, np.ndarray], \
                        K: Union[int, tuple, list, np.ndarray, None], \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        DN_L2_int_x: int, DN_L2_int_y: int, DN_L2_int_t: int, \
                        regression: bool=True
                        ) -> tuple:
        
        # If multiple element solver is used, the K parameter must be provided'
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided when using multi element solver.")
        
        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        

        # Convert the N list to np.ndarray
        if isinstance(N, int):
            N = np.array([N])
        elif isinstance(N, list):
            N = np.array(N)
        elif isinstance(N, tuple):
            N = np.array(N)  
        elif isinstance(N, range):
            N = np.array(list(N))
        
        # Convert the K list to np.ndarray
        if isinstance(K, int):
            K = np.array([K])
        elif isinstance(K, list):
            K = np.array(K)
        elif isinstance(K, tuple):
            K = np.array(K)
        elif isinstance(K, range):
            K = np.array(list(K))

        # Prelocate memory
        eL2 = np.zeros(shape = (len(K), len(N)) if K is not None else (1, len(N)), dtype=float)
        eLinf = np.zeros(shape = (len(K), len(N)) if K is not None else (1, len(N)), dtype=float)

        # Compute the array of L2s corresponding to each N-option
        print("Computing L2 errors (by definition) for...")

        for i in range(eL2.shape[0]):
            for j in range(eL2.shape[1]):
                n = N[j]
                k = K[i] if K is not None else None

                print(f"N = {n}, K = {k}")

                l2, linf = self.__compute_L2Linf_zw_h_def(dual, n, k, \
                                                n + DN_int_x, n + DN_int_y, \
                                                n + DN_int_t, n + DN_L2_int_x, n + DN_L2_int_y, n + DN_L2_int_t, \
                                                verbose=False)

                eL2[i, j] = l2
                eLinf[i, j] = linf

                print(f"N = {n}, K = {k}: eL2 = {l2}\n")
        
        # If the number of points in K or N directions are at least 2, perform regression on each dimension
        if regression:
            if len(N) > 1:
                reg_arr_L2 = np.empty(len(K), dtype=object)
                reg_arr_Linf = np.empty(len(K), dtype=object)

                for i in range(len(K)):
                    reg_arr_L2[i] = LinearRegression().fit(N.reshape(-1, 1), np.log10(eL2[i, :]))
                    reg_arr_Linf[i] = LinearRegression().fit(N.reshape(-1, 1), np.log10(eLinf[i, :]))


                return eL2, eLinf, reg_arr_L2, reg_arr_Linf

        return eL2, eLinf, None, None

    # Public method for L2 error for div
    def compute_L2Linf_div_h(self, \
                        N: Union[int, tuple, list, np.ndarray], \
                        K: Union[int, tuple, list, np.ndarray, None], \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        DN_L2_int_x: int, DN_L2_int_y: int, DN_L2_int_t: int, \
                        regression: bool=True
                        ) -> tuple:
        
        # If multiple element solver is used, the K parameter must be provided'
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided when using multi element solver.")
        
        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        

        # Convert the N list to np.ndarray
        if isinstance(N, int):
            N = np.array([N])
        elif isinstance(N, list):
            N = np.array(N)
        elif isinstance(N, tuple):
            N = np.array(N)  
        elif isinstance(N, range):
            N = np.array(list(N))
        
        # Convert the K list to np.ndarray
        if isinstance(K, int):
            K = np.array([K])
        elif isinstance(K, list):
            K = np.array(K)
        elif isinstance(K, tuple):
            K = np.array(K)
        elif isinstance(K, range):
            K = np.array(list(K))

        # Prelocate memory
        eL2 = np.zeros(shape = (len(K), len(N)) if K is not None else (1, len(N)), dtype=float)
        eLinf = np.zeros(shape = (len(K), len(N)) if K is not None else (1, len(N)), dtype=float)

        # Compute the array of L2s corresponding to each N-option
        print("Computing L2, Linf errors (by definition) for...")

        for i in range(eL2.shape[0]):
            for j in range(eL2.shape[1]):
                n = N[j]
                k = K[i] if K is not None else None

                print(f"N = {n}, K = {k}")

                l2, linf = self.__compute_L2Linf_div_h_def(n, k, \
                                                n + DN_int_x, n + DN_int_y, \
                                                n + DN_int_t, n + DN_L2_int_x, n + DN_L2_int_y, n + DN_L2_int_t, \
                                                verbose=False)

                eL2[i, j] = l2
                eLinf[i, j] = linf

                print(f"N = {n}, K = {k}: eL2 = {l2}, eLinf = {linf}\n")

        return eL2, eLinf

    # Public method for condition number
    def compute_condition_number(self, \
                        N: Union[int, tuple, list, np.ndarray], \
                        K: Union[int, tuple, list, np.ndarray, None], \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        DN_L2_int_x: int, DN_L2_int_y: int, DN_L2_int_t: int, \
                        mode: Literal['min', 'avg', 'max']='max' \
                        ) -> np.ndarray:
        
        # If multiple element solver is used, the K parameter must be provided'
        if self.type == 'ME' and K is None:
            raise ValueError("K parameter must be provided when using multi element solver.")
        
        # If single element solver is used, the K parameter must be None
        if self.type == 'SE' and K is not None:
            raise ValueError("K parameter must be left at None when using single element solver.")
        

        # Convert the N list to np.ndarray
        if isinstance(N, int):
            N = np.array([N])
        elif isinstance(N, list):
            N = np.array(N)
        elif isinstance(N, tuple):
            N = np.array(N)  
        elif isinstance(N, range):
            N = np.array(list(N))
        
        # Convert the K list to np.ndarray
        if isinstance(K, int):
            K = np.array([K])
        elif isinstance(K, list):
            K = np.array(K)
        elif isinstance(K, tuple):
            K = np.array(K)
        elif isinstance(K, range):
            K = np.array(list(K))

        # Prelocate memory
        CN = np.zeros(shape = (len(K), len(N)) if K is not None else (1, len(N)), dtype=float)

        # Compute the array of L2s corresponding to each N-option
        print("Computing condition numbers for...")

        for i in range(CN.shape[0]):
            for j in range(CN.shape[1]):
                n = N[j]
                k = K[i] if K is not None else None

                print(f"N = {n}, K = {k}")

                cn = self.__compute_condition_number(n, k, \
                                                n + DN_int_x, n + DN_int_y, \
                                                n + DN_int_t, n + DN_L2_int_x, n + DN_L2_int_y, n + DN_L2_int_t, \
                                                mode=mode, \
                                                verbose=False)

                CN[i, j] = cn

                print(f"N = {n}, K = {k}: condition number = {cn}\n")

        return CN

    # Deprecated method
    def __depr_compute_L2_zw_h_M0(self, \
                        dual: bool, \
                        N: int, \
                        N_int: Union[None, int], \
                        zw_exact: Callable, \
                        verbose: bool=False \
                        ) -> float:

        # Create auxiliary solver: the same args as the top_solver but update orders
        aux_kwargs = self.top_solver.kwargs
        aux_kwargs["N"] = N
        aux_kwargs["N_int_x"] = N
        aux_kwargs["N_int_y"] = N
        aux_kwargs["N_int_t"] = N

        aux_solver = Solver(self.top_solver.problem_id, self.top_solver.sparse, verbose=verbose, **aux_kwargs)
        metric = Metric(N_int, N_int, N_int, N_int, aux_solver.problem.c, aux_solver.problem.r, aux_solver.problem.d_map, aux_solver.problem.t_map)

        # Retrieve mappings
        d_map = metric.d_map
        t_map = metric.t_map

        # Generate unit integration grid
        xi_int_grid, tau_int_grid, eta_int_grid = np.meshgrid(metric.GL_int_x, metric.GL_int_y, metric.GL_int_t)
        mesh_xieta_int = np.meshgrid(metric.GL_int_x, metric.GL_int_y)

        x_int_grid = d_map.x(xi_int_grid, eta_int_grid)
        y_int_grid = d_map.y(xi_int_grid, eta_int_grid)
        t_int_grid = t_map.t(tau_int_grid)

        # Solve with solver
        aux_solver.solve_system(verbose)

        if dual:
            z_h = aux_solver.reconstruct_w(mesh_xieta_int, t_map.t(metric.GL_int_t), verbose=False)   # solve for w DOFs
        else:
            z_h = aux_solver.reconstruct_z(mesh_xieta_int, t_map.t(metric.GL_int_t), verbose=False)   # solve for z DOFs
        

        # Evaluate the exact solution at integration nodes
        zw_ex_evaluated = zw_exact(x_int_grid, y_int_grid, t_int_grid)
        
        # get M0
        M0 = metric.M_0(False)

        # Error vector
        e = (zw_ex_evaluated - z_h).reshape(-1)

        # Integrate the error squared
        I = e.T @ M0 @ e

        # Return the value sqrt
        return np.sqrt(I)

    def compute_energy(self, t: Union[float, np.ndarray, list], \
                        N_int_x: Union[None, int]=None, N_int_y: Union[None, int]=None, \
                        verbose: bool=False):
        
        # Inform
        print("Computing total energy...\n")

        # Get the top problem
        top_problem = self.top_problem
        top_solver = self.top_solver

        # Check if the system is solved. If not -- solve it
        if self.type == 'SE':
            if top_solver.get_solution() is None:
                top_solver.solve_system(verbose=False)
        else:
            if top_solver.solver_chain is None:
                top_solver.solve(verbose=False)

        # Assign appropriate integration orders
        if N_int_x is None:
            N_int_x = top_problem.N_int_x

        if N_int_y is None:
            N_int_y = top_problem.N_int_y

        # Process the time slices
        if not (isinstance(t, float) or isinstance(t, np.ndarray) or isinstance(t, list)):
            raise ValueError("The 't' arguments (time slices) must be either of type float, np.ndarray or list.")
        
        if isinstance(t, float):
            t = np.array([t])
        elif isinstance(t, list):
            t = np.array(t)

        # Get the integration weights and integration points
        GL_int_x, weights_int_x = lobatto_quad(N_int_x)
        GL_int_y, weights_int_y = lobatto_quad(N_int_y)

        # Make the reconstruction integration mesh grid
        mesh_grid_xieta_rec_int = np.meshgrid(GL_int_x, GL_int_y)

        # Reconstruct physical quantities
        sigma_kx = top_solver.reconstruct_sigma_kx(mesh_grid_xieta_rec_int, t, verbose=verbose)
        sigma_ky = top_solver.reconstruct_sigma_ky(mesh_grid_xieta_rec_int, t, verbose=verbose)
        pi = top_solver.reconstruct_pi(mesh_grid_xieta_rec_int, t, verbose=verbose)

        # Compute the total energy at all time slices
        e_pot = 0.5 * (sigma_kx ** 2 + sigma_ky ** 2)
        e_kin = 0.5 * pi ** 2
        e_tot = e_pot + e_kin

        # Integrate the energy for each time slice
        e_pot = np.einsum('ijk,k,j->i', e_pot, weights_int_x, weights_int_y)
        e_kin = np.einsum('ijk,k,j->i', e_kin, weights_int_x, weights_int_y)
        e_tot = np.einsum('ijk,k,j->i', e_tot, weights_int_x, weights_int_y)

        # All done
        return e_pot, e_kin, e_tot

    def plot_abs_spat_ptwise_error_zw_h(self, \
                        dual: bool, \
                        t: float, \
                        N: int, K: int, \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        N_res_x: int, N_res_y: int, \
                        adjust_font: bool=False, \
                        show: bool=True, save_name: Union[str, None]=None):
        
        # Adjust the font
        if adjust_font:
            PlotterHelper.adjust_font()

        # Get colors, markers and markersize
        my_colors, _, _ = PlotterHelper.get_colors_markers_markersize()

        # Get the error
        abs_ptwse_err = self.__compute_abs_spat_ptwise_error_zw_h(dual, t, N, K, DN_int_x, DN_int_y, DN_int_t, N_res_x, N_res_y)

        # Convert to log10
        abs_ptwse_err = np.log10(abs_ptwse_err) 

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=PlotterHelper.get_figsize())

        # Plot the mesh
        N = self.top_problem.N
        self.top_problem.d_map.plot(N, N, color='white', linewidth=0.5, linestyle='--', alpha=0.3, ax=ax)

        # Create the color map
        cvals  = [-8.914516222598609, -1.1799912639273973]
        colors = [my_colors[0], my_colors[3]]

        norm=plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        # min max for zw, t 1, 2
        # min = -8.914516222598609
        # max = -1.1799912639273973
        # print(f"min = {np.min(abs_ptwse_err)}, max = {np.max(abs_ptwse_err)}")

        # Plot the 2D array using imshow
        cax = ax.imshow(abs_ptwse_err, cmap=cmap, norm=norm, interpolation='none', extent=[-1, 1, -1, 1])

        # Create a divider for the existing axis
        # divider = make_axes_locatable(ax)
        # # Append a new axis to the right of the current axis, for the colorbar
        # cax_cb = divider.append_axes("right", size="5%", pad=0.05)

        # ticks = np.linspace(np.min(abs_ptwse_err), np.max(abs_ptwse_err), 4)
        # ticks = np.ceil(ticks)
        # ticks.astype(int)

        cbar = fig.colorbar(cax, fraction=0.05)  # Add a colorbar to show the color scale
        cbar.set_label(r'$\log_{10} \left( \left| w^{ex} - w^{h} \right| \right)$')
        # cbar.set_ticks(ticks)


        # Adding titles and labels
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

        # Tight layout
        plt.tight_layout()

        # Showing and saving
        if save_name is not None:
            plt.savefig('res//log_e_w//' + save_name)
        if show:
            plt.show()

    def plot_abs_spat_ptwise_error_pi_h(self, \
                        dual: bool, \
                        t: float, \
                        N: int, K: int, \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        N_res_x: int, N_res_y: int, \
                        adjust_font: bool=False, \
                        show: bool=True, save_name: Union[str, None]=None):
        
        # Adjust the font
        if adjust_font:
            PlotterHelper.adjust_font()

        # Get colors, markers and markersize
        my_colors, _, _ = PlotterHelper.get_colors_markers_markersize()

        # Get the error
        abs_ptwse_err = self.__compute_abs_spat_ptwise_error_pi_h(dual, t, N, K, DN_int_x, DN_int_y, DN_int_t, N_res_x, N_res_y)

        # Convert to log10
        abs_ptwse_err = np.log10(abs_ptwse_err) 

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=PlotterHelper.get_figsize())

        # Plot the mesh
        N = self.top_problem.N
        self.top_problem.d_map.plot(N, N, color='white', linewidth=0.5, linestyle='--', alpha=0.3, ax=ax)

        # Create the color map
        cvals  = [-7.1277045320496395, 0.44079439027450185]
        colors = [my_colors[0], my_colors[3]]

        # min max for pi
        # min = -7.1277045320496395
        # max = 0.44079439027450185
        # print(f"min = {np.min(abs_ptwse_err)}, max = {np.max(abs_ptwse_err)}")

        norm=plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        # Plot the 2D array using imshow
        cax = ax.imshow(abs_ptwse_err, cmap=cmap, norm=norm, interpolation='none', extent=[-1, 1, -1, 1])

        # Create a divider for the existing axis
        # divider = make_axes_locatable(ax)
        # # Append a new axis to the right of the current axis, for the colorbar
        # cax_cb = divider.append_axes("right", size="5%", pad=0.05)

        # ticks = np.linspace(np.min(abs_ptwse_err), np.max(abs_ptwse_err), 4)
        # ticks = np.ceil(ticks)
        # ticks.astype(int)

        cbar = fig.colorbar(cax, fraction=0.05)  # Add a colorbar to show the color scale
        cbar.set_label(r'$\log_{10} \left( \left| {\pi}^{ex} - {\pi}^{h} \right| \right)$')
        # cbar.set_ticks(ticks)


        # Adding titles and labels
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

        # Tight layout
        plt.tight_layout()

        # Showing and saving
        if save_name is not None:
            plt.savefig('res//log_e_pi//' + save_name)
        if show:
            plt.show()

    def plot_L2Linf_zw_h(self, dual: bool, \
                        N: Union[int, tuple, list, np.ndarray], \
                        K: Union[int, tuple, list, np.ndarray, None], \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        DN_L2_int_x: int, DN_L2_int_y: int, DN_L2_int_t: int, \
                        regression: bool=True, \
                        adjust_font: bool=False, \
                        show: bool=True, save_name: Union[str, None]=None):
        

        # Convert the K list to np.ndarray
        if isinstance(K, int):
            K = np.array([K])
        elif isinstance(K, list):
            K = np.array(K)
        elif isinstance(K, tuple):
            K = np.array(K)
        elif isinstance(K, range):
            K = np.array(list(K))

        # Font options
        if adjust_font:
            PlotterHelper.adjust_font()

        colors, markers, markersize = PlotterHelper.get_colors_markers_markersize()
        

        # Compute the L2 errors
        eL2, eLinf, reg_arr_L2, reg_arr_Linf = self.compute_L2Linf_zw_h(dual, N, K, DN_int_x, DN_int_y, DN_int_t, DN_L2_int_x, DN_L2_int_y, DN_L2_int_t, regression)

        # array to store legend handles
        im = [0 for i in range(2 * len(K))]

        # Adding legend inside a box
        fig, ax = plt.subplots(1, figsize=PlotterHelper.get_figsize())

        ax.set_yscale('log')

        # Plotting
        for i in range(eL2.shape[0]):
            im[i], = ax.plot(N, eL2[i, :], color=colors[i], marker=markers[i], markersize=markersize)

        for i in range(eLinf.shape[0]):
            im[i + len(K)], = ax.plot(N, eLinf[i, :], color=colors[i], marker=markers[i], linestyle='--', markersize=markersize)

        # Set ticks on the x-axis
        ax.set_xticks(N)

        # Axes labels
        ax.set_xlabel('$N$')
        ax.set_ylabel(r'$\left \Vert w \right \Vert ^e_{L^p_{\Omega \times T}}$')

        # Legend

        # create blank rectangle
        extra = Rectangle((0, 0), 2, 2, fc="w", fill=False, edgecolor='k', linewidth=0)
        empty = [""]

        if adjust_font:
            label_key = [r'$K \backslash p$']   
            label_col = [r'$\textcolor{white}{|||}$' + f'${i}$' for i in K]   
            label_row = [r'$\textcolor{white}{|||} 2 \textcolor{white}{|||}$', r'$\textcolor{white}{||} \infty \textcolor{white}{||}$']
        else:
            label_key = [r'$K \backslash p$']   
            label_col = [f'${i}$' for i in K]   
            label_row = [r'$2$', r'$\infty$']       

        legend_handle = [*[extra for i in range(2 + len(K))], *im[0 : len(K)], extra, *im[len(K):]]
        legend_labels = np.concatenate([label_key, label_col, [label_row[0]], len(K) * empty, [label_row[1]], len(K) * empty])


        legend = ax.legend
        legend(legend_handle, legend_labels, \
               borderpad=0.5, columnspacing = 0.4, handletextpad = -2, \
               loc='best', shadow=False, ncol=3, frameon=True, framealpha=1, facecolor=None, edgecolor='black').get_frame().set_boxstyle('square', pad=0.2)

        plt.tight_layout()

        if regression:
            a = reg_arr_L2[-1].coef_[0]

            # Perform regression on the highest K
            ax.plot(N, 10 ** reg_arr_L2[-1].predict(np.array(N).reshape(-1, 1)), color=colors[-1], linestyle='-.')
            print(f"slope = {a}")
            

        # Showing and saving
        if save_name is not None:
            plt.savefig('res//l2linf//' + save_name)
        if show:
            plt.show()

    def plot_L2Linf_div_h(self, \
                        N: Union[int, tuple, list, np.ndarray], \
                        K: Union[int, tuple, list, np.ndarray, None], \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        DN_L2_int_x: int, DN_L2_int_y: int, DN_L2_int_t: int, \
                        regression: bool=True, \
                        adjust_font: bool=False, \
                        show: bool=True, save_name: Union[str, None]=None):
        
        
        # Convert the K list to np.ndarray
        if isinstance(K, int):
            K = np.array([K])
        elif isinstance(K, list):
            K = np.array(K)
        elif isinstance(K, tuple):
            K = np.array(K)
        elif isinstance(K, range):
            K = np.array(list(K))

        # Font options
        if adjust_font:
            PlotterHelper.adjust_font()

        colors, markers, markersize = PlotterHelper.get_colors_markers_markersize()

        # Compute the L2 errors
        eL2, eLinf = self.compute_L2Linf_div_h(N, K, DN_int_x, DN_int_y, DN_int_t, DN_L2_int_x, DN_L2_int_y, DN_L2_int_t, regression)

        # array to store legend handles
        im = [0 for i in range(2 * len(K))]

        # Adding legend inside a box
        fig, ax = plt.subplots(1, figsize=PlotterHelper.get_figsize())

        ax.set_yscale('log')

        # Plotting
        for i in range(eL2.shape[0]):
            im[i], = ax.plot(N, eL2[i, :], color=colors[i], marker=markers[i], markersize=markersize)

        for i in range(eLinf.shape[0]):
            im[i + len(K)], = ax.plot(N, eLinf[i, :], color=colors[i], marker=markers[i], linestyle='--', markersize=markersize)

        # Set ticks on the x-axis
        ax.set_xticks(N)

        # Axes labels
        ax.set_xlabel('$N$')
        ax.set_ylabel(r'$\left \Vert \Delta w - \frac{\partial ^ 2 w}{\partial t ^ 2} \right \Vert ^ e_{L^p_{\Omega \times T}}$')
        
        # Legend

        # create blank rectangle
        extra = Rectangle((0, 0), 2, 2, fc="w", fill=False, edgecolor='k', linewidth=0)
        empty = [""]

        if adjust_font:
            label_key = [r'$K \backslash p$']   
            label_col = [r'$\textcolor{white}{|||}$' + f'${i}$' for i in K]   
            label_row = [r'$\textcolor{white}{|||} 2 \textcolor{white}{|||}$', r'$\textcolor{white}{||} \infty \textcolor{white}{||}$']
        else:
            label_key = [r'$K \backslash p$']   
            label_col = [f'${i}$' for i in K]   
            label_row = [r'$2$', r'$\infty$'] 

        legend_handle = [*[extra for i in range(2 + len(K))], *im[0 : len(K)], extra, *im[len(K):]]
        legend_labels = np.concatenate([label_key, label_col, [label_row[0]], len(K) * empty, [label_row[1]], len(K) * empty])


        legend = ax.legend
        legend(legend_handle, legend_labels, \
               borderpad=0.5, columnspacing = 0.4, handletextpad = -2, \
               loc='best', shadow=False, ncol=3, frameon=True, framealpha=1, facecolor=None, edgecolor='black').get_frame().set_boxstyle('square', pad=0.2)

        plt.tight_layout()

        # Showing and saving
        if save_name is not None:
            plt.savefig('res//l2linf//' + save_name)
        if show:
            plt.show()

    def plot_energy(self, t: Union[np.ndarray, list, range, tuple], \
                        N_int_x: Union[None, int]=None, N_int_y: Union[None, int]=None, \
                        adjust_font: bool=False, \
                        show: bool=True, save_name: Union[str, None]=None, \
                        verbose: bool=False):
        
        # Convert the t list to np.ndarray
        if isinstance(t, int) or isinstance(t, float):
            raise ValueError("The 't' arguments (time slices) must be either of type np.ndarray, list, range or tuple.")
        elif isinstance(t, list):
            t = np.array(t)
        elif isinstance(t, tuple):
            t = np.array(t)
        elif isinstance(t, range):
            t = np.array(list(t))

        # Font options
        if adjust_font:
            PlotterHelper.adjust_font()

        colors, _, _ = PlotterHelper.get_colors_markers_markersize()

        # Get the energy
        e_pot, e_kin, e_tot = self.compute_energy(t, N_int_x, N_int_y, verbose)
        e_tot_ex = self.top_problem.e_tot_exact
        
        # Adding legend inside a box
        fig, ax = plt.subplots(1, figsize=PlotterHelper.get_figsize())
        
        # Plotting
        ax.plot(t, e_tot, label=r'$E^h$', color=colors[0], zorder=2)

        ax.set_xlabel('$t$')

        # If exact energy is available
        if e_tot_ex is not None:
            ax.plot([min(t), max(t)], [e_tot_ex, e_tot_ex], label=r'$E^{ex}$', linestyle='-', color=colors[1], zorder=1)

            # Get the max y and min y
            y_min = min(e_tot_ex, min(e_tot))
            y_max = max(e_tot_ex, max(e_tot))

        # Get the max and min x, y
        y_min = min(e_tot)
        y_max = max(e_tot)
        x_min = min(t)
        x_max = max(t)

        legend = ax.legend
        legend(loc='upper right', shadow=False, frameon=True, framealpha=1, facecolor=None, edgecolor='black').get_frame().set_boxstyle('square', pad=0.2)

        # Get solver
        solver = self.top_solver

        # Get time divisions
        if isinstance(solver, MESolver):
            t_div = np.linspace(solver.t0, solver.tf, solver.K + 1)

            # Plot vertical lines, element boundaries
            for i in range(len(t_div)):
                ax.axvline(x=t_div[i], color=colors[2], linestyle='--', zorder=0, linewidth=0.5)

        # ax.grid(True, color='0.8')
        # ax.set_xlim(left=x_min, right=x_max)
        # ax.set_ylim(bottom=y_min, top=y_max)

        # bbox_to_anchor moves the legend box relative to the location specified in loc
        # In this case, it moves it down by 0.1 relative to upper left, effectively placing it below the plot

        plt.tight_layout()

        if  save_name is not None:
            plt.savefig('res//energy//' + save_name)

        if show:
            plt.show()

    def plot_condition_number(self, \
                        N: Union[int, tuple, list, np.ndarray], \
                        K: Union[int, tuple, list, np.ndarray, None], \
                        DN_int_x: int, DN_int_y: int, DN_int_t: int, \
                        DN_L2_int_x: int, DN_L2_int_y: int, DN_L2_int_t: int, \
                        adjust_font: bool=False, \
                        mode: Literal['min', 'avg', 'max']='max', \
                        show: bool=True, save_name: Union[str, None]=None):
        
        # Font options
        if adjust_font:
            plt.rcParams.update({
                "text.usetex": True,  # Use TeX for rendering
                "font.family": "serif",  # Use serif font
                "font.serif": ["cmr10"],  # Set the CMR font family and size
                "font.size": 10,  # Set the default font size
                "axes.formatter.use_mathtext": True  # Enable mathtext mode
            })

        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'h', 'H', '+', 'x', '|', '_']
        colors = ["#CC6677", "#999933", "#117733", "#332288"][::-1]

        # Compute the L2 errors
        CN = self.compute_condition_number(N, K, DN_int_x, DN_int_y, DN_int_t, DN_L2_int_x, DN_L2_int_y, DN_L2_int_t, mode=mode)

        # Adding legend inside a box
        fig, ax = plt.subplots(1, figsize=PlotterHelper.get_figsize())

        ax.set_yscale('log')

        # Plotting
        for i in range(CN.shape[0]):
            ax.plot(N, CN[i, :], label=f'$K = {K[i]}$' if K is not None else '$K = None$', color=colors[i], marker=markers[i])

        # Set ticks on the x-axis
        ax.set_xticks(N)

        # Axes labels
        ax.set_xlabel('$N$')
        ax.set_ylabel(r'$\kappa$')
        
        # Legend
        # loc: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        legend = ax.legend
        legend(loc='best', shadow=False, ncol=2, frameon=True, framealpha=1, facecolor=None, edgecolor='black').get_frame().set_boxstyle('square', pad=0.2)

        plt.tight_layout()

        # Showing and saving
        if save_name is not None:
            plt.savefig('res//' + save_name)
        if show:
            plt.show()

    def plot_exponent_c(self, \
                        adjust_font: bool=False, \
                        show: bool=True, save_name: Union[str, None]=None):
        
    
        # Font options
        if adjust_font:
            PlotterHelper.adjust_font()

        colors, markers, markersize = PlotterHelper.get_colors_markers_markersize()

        # Get the exponents and c
        df = pd.read_excel('res//c_exponent//exponents.xlsx', sheet_name='exponents')
        table_df = df.iloc[4:, 12:].values
        c_array = np.array(table_df[:-1, 0], dtype=float)
        exponent_array = np.array(table_df[:-1, 1], dtype=float)


        # Adding legend inside a box
        fig, ax = plt.subplots(1, figsize=PlotterHelper.get_figsize())

        ax.scatter(c_array, exponent_array, color=colors[0], marker=markers[0], s=markersize)
        ax.plot(c_array[:-2], exponent_array[:-2], color=colors[0], linestyle='-')
        ax.plot(c_array[-3:], exponent_array[-3:], color=colors[0], linestyle='--')
        
        ax.set_xticks(c_array)
        # Rotate x-ticks
        plt.xticks(rotation=45)

        # Axes labels
        ax.set_xlabel('$c$')
        ax.set_ylabel(r'$a$')

        # legend = ax.legend
        # legend(loc='best', shadow=False, ncol=2, frameon=True, framealpha=1, facecolor=None, edgecolor='black').get_frame().set_boxstyle('square', pad=0.2)


        plt.tight_layout()

        # Showing and saving
        if save_name is not None:
            plt.savefig('res//c_exponent//' + save_name)
        if show:
            plt.show()

    def plot_meshes(self, \
                        adjust_font: bool=False, \
                        show: bool=True, save_name: bool=False):
        
        # Font options
        if adjust_font:
            PlotterHelper.adjust_font()

        colors, markers, markersize = PlotterHelper.get_colors_markers_markersize()


        # Adding legend inside a box
        fig1, ax1 = plt.subplots(1, figsize=PlotterHelper.get_figsize())
        fig3, ax3 = plt.subplots(1, figsize=PlotterHelper.get_figsize())
        fig4, ax4 = plt.subplots(1, figsize=PlotterHelper.get_figsize())

        # Get maps
        d_map1 = StandardDomainMapping("crazy_mesh", c=0.1)
        d_map3 = StandardDomainMapping("crazy_mesh", c=0.3)
        d_map4 = StandardDomainMapping("crazy_mesh", c=0.4)

        # Plot the meshes
        d_map1.plot(9, 9, 100, False, None, 'k', None, False, ax1)
        d_map3.plot(9, 9, 100, False, None, 'k', None, False, ax3)
        d_map4.plot(9, 9, 100, False, None, 'k', None, False, ax4)
        

        # Axes labels
        ax1.set_xlabel(r'$\xi$')
        ax1.set_ylabel(r'$\eta$')

        ax3.set_xlabel(r'$\xi$')

        ax4.set_xlabel(r'$\xi$')

        fig1.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()

        # Showing and saving
        if save_name:
            fig1.savefig('res//mesh//mesh_c1.pdf')
            fig3.savefig('res//mesh//mesh_c3.pdf')
            fig4.savefig('res//mesh//mesh_c4.pdf')
        if show:
            plt.show()


if __name__ == "__main__":
    N0 = 12
    K0 = 4

    s = Solver(problem_id=1, sparse=False, N=N0, N_int_x=N0, N_int_y=N0, N_int_t=N0, \
               t_map=StandardTimeMapping("linear", t_begin=-1, t_end=1), d_map = StandardDomainMapping("crazy_mesh", c=0.0), \
               verbose=False)
    # s.print_problem()
    

    mes = MESolver(problem_id=1, K=K0, sparse=False, N=N0, N_int_x=N0, N_int_y=N0, N_int_t=N0, \
                    t_map=StandardTimeMapping("linear", t_begin=0., t_end=2.),
                    d_map = StandardDomainMapping("crazy_mesh", c=0.1), verbose=False)
    # mes.print_problem()

    a = Analyzer(mes)

    a.plot_L2Linf_zw_h(False, \
                    range(1, 11), [1, 2, 3], \
                    2, 2, 2, \
                    3, 3, 3, \
                    adjust_font=True, \
                    regression=False, \
                    show=True, \
                    save_name='L2Linf_errors_c1.ps')
    
    # a.plot_exponent_c(adjust_font=True, show=True, save_name='exponent_c.pdf')

    # a.plot_energy(np.linspace(0, 2, 300), \
    #                 N_int_x=N0+3, N_int_y=N0+3, \
    #                 adjust_font=True, show=True, save_name='energy_k4_c1.pdf')

    # a.plot_abs_spat_ptwise_error_pi_h(False, \
    #                                     1., \
    #                                     10, 3, \
    #                                     2, 2, 2, \
    #                                     200, 200, \
    #                                     adjust_font=True, \
    #                                     show=True, save_name='log_e_pi_k3_t1_c2_alpha.pdf')

    # a.plot_meshes(adjust_font=True, show=True, save_name=True)

    # a.plot_L2Linf_div_h(range(2, 11), [1, 2, 3], \
    #                 2, 2, 2, \
    #                 3, 3, 3, \
    #                 adjust_font=True, \
    #                 regression=False,
    #                 save_name='L2Linf_errors_div_c1.ps')

    # a.plot_condition_number(range(2, 7), [1, 2, 3], \
    #                 2, 2, 2, \
    #                 3, 3, 3, \
    #                 adjust_font= True, \
    #                 save_name='condition_number_c1.pdf')