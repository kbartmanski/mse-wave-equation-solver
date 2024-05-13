import numpy as np
import inspect
import textwrap
import time
from mapping import DomainMapping, TimeMapping, StandardDomainMapping, StandardTimeMapping
from components import UnitTopology, Metric
from reconstruction import Reconstruction
from plotter import Plotter3D, Animator
from typing import Callable, Tuple, Union, Any
from types import NoneType
from scipy.sparse import csr_matrix, linalg

class Reconstructible(object):

    """
    An abstract class covering an object that can be reconstructed. Contains 7 abstract methods
    """

    # Abstract methods
    def reconstruct_w(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        raise NotImplementedError("This abstract method has not been implemented yet.")

    def reconstruct_z(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        raise NotImplementedError("This abstract method has not been implemented yet.")

    def reconstruct_sigma_kx(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        raise NotImplementedError("This abstract method has not been implemented yet.")

    def reconstruct_sigma_ky(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        raise NotImplementedError("This abstract method has not been implemented yet.")

    def reconstruct_sigma_ky(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        raise NotImplementedError("This abstract method has not been implemented yet.")

    def reconstruct_pi(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        raise NotImplementedError("This abstract method has not been implemented yet.")
    
    def reconstruct_div(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        raise NotImplementedError("This abstract method has not been implemented yet.")

class Problem(object):
    def __init__(   self, id: int, description: str,\
                    N: int, N_int_x:int, N_int_y:int, N_int_t:int,\
                    c :Callable, r: Callable,\
                    d_map: DomainMapping, t_map: TimeMapping,\
                    bc: Tuple[str, Any]):
        
        # Distribute attributes
        self.id = id
        self.description: str = description
        self.N = N
        self.N_int_x = N_int_x
        self.N_int_y = N_int_y
        self.N_int_t = N_int_t
        self.c = c
        self.r = r
        self.d_map = d_map
        self.t_map = t_map
        self.bc = bc            # is tuple with TWO entries (*, *)
        self.zw_exact = None
        self.e_tot_exact = None

        # For multiple elements only
        self.K = None
    
    def __str__(self) -> str:

        if self.id == 0:
            return self.description
        
        else:
            out: str = "--------------------\nProblem " + str(self.id) + ":\n"
            out += self.description + "\n"
            out += f"{'' if self.K is None else f'K = {self.K}, '}"
            out += f"N = {self.N}, N_int_x = {self.N_int_x}, N_int_y = {self.N_int_y}, N_int_t = {self.N_int_t}\n\n"
            out += f"{textwrap.dedent(inspect.getsource(self.c))}{textwrap.dedent(inspect.getsource(self.r))}\n"
            out += str(self.d_map) + "\n" + str(self.t_map) + "\n\n"
            out += f"Boundary conditions: {self.bc[0]}\n{textwrap.dedent(inspect.getsource(self.bc[1]))}---------------------\n"

            return out

class ValidProblemID(object):
    valid_problem_id = [0, 1]

class Solver(ValidProblemID, Reconstructible):

    """
    Single element solver class.
    """

    def __init__(self, problem_id: int, sparse: bool, verbose: bool=True, **kwargs) -> None:

        if problem_id not in Solver.valid_problem_id:
            raise ValueError(problem_id + " not valid. See valid ids: " + str(Solver.valid_problem_id))
        
        self.problem_id = problem_id
        self.sparse = sparse
        self.verbose = verbose
        self.kwargs = kwargs

        self.problem: Problem = self.get_problem()
        
        # may change the metric for reconstruction (only overwrite if using ME!)
        self.overwritten_metric: Union[Metric, None] = None

        self.__lhs = None
        self.__rhs = None
        self.__solution = None

    def get_problem(self) -> Problem:
        
        if self.problem_id == 0:
            # A 'blank' problem. All attributes initialized to None

            # any **kwargs should not have been passed if choosing the phantom problem
            if self.kwargs:
                raise ValueError("For the blank problem no kwargs should be passed.")

            kwargs = {
            "id": 0, "description": "A blank problem. Phantom (None) attributes.",\
            "N": None, "N_int_x": None, "N_int_y": None, "N_int_t": None, \
            "c": None, \
            "r": None, \
            "d_map": None, \
            "t_map": None, \
            "bc": (None, None)
            }

            return Problem(**kwargs)

        elif self.problem_id == 1:
                
            c = lambda x, y, t: 0 * x + 0 * y + 0 * t + 1.
            r = lambda x, y, t: 0 * x + 0 * y + 0 * t + 1.
            w0 = lambda x, y: np.cos(np.pi/2*x)*np.cos(np.pi/2*y)

            default_kwargs = {
            "id": 1, "description": "Solve WE on a SINGLE element with the properties as follows.",\
            "N": 1, "N_int_x": 1, "N_int_y": 1, "N_int_t": 1, \
            "c": c, \
            "r": r, \
            "d_map": StandardDomainMapping("crazy_mesh", c=0.1), \
            "t_map": StandardTimeMapping("linear", t_begin=0., t_end=2.), \
            "bc": ("all zero, except initial displ. ", w0)
            }

            # Overwrite
            for key, value in self.kwargs.items():
                default_kwargs[key] = value

            problem = Problem(**default_kwargs)

            def exact_solution(x, y, t):
                c = 1
                Lx = 2
                Ly = 2
                
                n = 1
                m = 1

                omega = c * np.pi * np.sqrt((n / Lx) ** 2 + (m / Ly) ** 2)
                return np.cos(np.pi / 2 * x) * np.cos(np.pi / 2 * y) * np.cos(omega * t)

            # Assign exact solution                                                        
            problem.zw_exact = exact_solution

            # Assign exact energy
            problem.e_tot_exact = np.pi ** 2 / 4

            return problem
      
    def print_problem(self) -> None:
        print(self.problem)

    def overwrite_metric(self, metric: Metric) -> None:
        self.overwritten_metric = metric
    
    def get_unit_topology(self) -> UnitTopology:
        return UnitTopology(self.get_metric().N)

    def get_metric(self) -> Metric:
        
        # First check if the metric is not overwitten (if so: use it)
        if self.overwritten_metric is not None:
            return self.overwritten_metric
        
        else:
            # Retrieve features
            N = self.problem.N
            N_int_x = self.problem.N_int_x
            N_int_y = self.problem.N_int_y
            N_int_t = self.problem.N_int_t

            c = self.problem.c
            r = self.problem.r

            d_map = self.problem.d_map
            t_map = self.problem.t_map

        return Metric(N, N_int_x, N_int_y, N_int_t, c, r, d_map, t_map)

    def get_solution(self) -> Union[np.ndarray, None]:
        return self.__solution
    
    def get_condition_number(self) -> float:
        """Accessor method to get the condition number of the lhs matrix.

        Raises:
            ValueError: Raised when the solution is not yet computed. Must self.solve_system(args) the solver first.

        Returns:
            float: The condition number of the lhs matrix.
        """

        if self.__solution is None:
            raise ValueError("The solution is not computeted yet. Compute it by self.solve_system(args).\n")
        
        if self.sparse:
            return np.linalg.cond(csr_matrix.todense(self.__lhs))
        
        return np.linalg.cond(self.__lhs)
    
    def compute_rhs(self, verbose: bool) -> None:

        # Start the clock
        start_time = time.perf_counter()

        # Construct metric
        metric = self.get_metric()

        # Retrieve features
        N = metric.N
        d_map = metric.d_map

        # Construct topology
        utop = UnitTopology(N)

        # split into cases
        if self.problem_id == 1:
            
            # get the initial displacement function
            w0 = self.problem.bc[1]

            e_x = Metric._extend_repeat_matrix(metric.E_MATRIX_X, n = N, m = 1)
            e_y = Metric._extend_repeat_matrix(metric.E_MATRIX_Y, n = 1, m = N)

            T = Metric._tile_2_matrix(e_x, e_y)

            # Evaluate w0(x, y) at integration nodes:

            # Unit integration grid
            XI_INT, ETA_INT = np.meshgrid(metric.GL_int_x, metric.GL_int_y)
            XI_INT = XI_INT.T
            ETA_INT = ETA_INT.T

            # NOTE; definition: physical integration grid = transformed(unit integration grid)
            # Transformed physical integration grid 
            X_INT = d_map.x(XI_INT, ETA_INT)
            Y_INT = d_map.y(XI_INT, ETA_INT)

            W0 = w0(X_INT, Y_INT)

            # integrate [iint_unitdOmega f(x(xi, eta), y(xi, eta))e(xi)e(eta) dxi deta]
            I = np.einsum('iab,ab,a,b->i', T, W0, metric.weights_int_x, metric.weights_int_y)

            # Multiply with the initial inclusion matrix
            b = utop.N_2_0(verbose=False) @ I

            # rhs vector v
            v = np.hstack((np.zeros(N ** 3), b, np.zeros(2 * N ** 2 * (N + 1) + N ** 2))).T

            end_time = time.perf_counter()

            # report time
            if verbose:
                print(f"rhs construction: {end_time - start_time} seconds.\n")

            return v
        
        # In any other case (0 or not-covered id) raise an error
        else:
            raise ValueError("rhs cannot be computed due to invalid id. Check your problem definition.")

    def compute_lhs(self, verbose: bool, sparse: bool) -> None:

        # Start the clock
        start_time = time.perf_counter()

        # Construct metric
        metric = self.get_metric()

        # Retrieve features
        N = metric.N

        # Construct topology
        utop = UnitTopology(N)


        # split into cases
        if self.problem_id == 1 or self.problem_id == 0:
            E32x = utop.E_3_2_x(verbose)
            E32y = utop.E_3_2_y(verbose)
            E32t = utop.E_3_2_t(verbose)
            E32xy = np.hstack((E32x, E32y))

            N20 = utop.N_2_0(verbose)
            N2T = utop.N_2_T(verbose)

            Mr = metric.M_1_r(verbose)
            Mc00 = metric.M_1_c_00(verbose)
            Mc01 = metric.M_1_c_01(verbose)
            Mc10 = metric.M_1_c_01(verbose).T
            Mc11 = metric.M_1_c_11(verbose)
            Mc = np.block([[Mc00, Mc01],
                        [Mc10, Mc11]])

            row1 = np.hstack((np.zeros((E32t.shape[0], E32t.shape[0])), E32t, E32xy, np.zeros((E32t.shape[0], N2T.shape[1]))))
            row2 = np.hstack((E32t.T, Mr, np.zeros((Mr.shape[0], 2 * E32x.shape[1])), -N2T))
            row3 = np.hstack((-E32xy.T, np.zeros((Mc.shape[0], Mr.shape[1])), Mc, np.zeros((Mc.shape[0], N2T.shape[1]))))
            row4 = np.hstack((np.zeros((N20.shape[1], E32t.shape[0])), -N20.T, np.zeros((N20.shape[1], Mc.shape[1])), np.zeros((N20.shape[1], N2T.shape[1]))))

            if sparse:
                A_sp = csr_matrix(np.vstack((row1, row2, row3, row4)))

                end_time = time.perf_counter()

                # report time
                if verbose:
                    print(f"lhs sparse computation: {end_time - start_time} seconds.\n")

                return A_sp
            
            A = np.vstack((row1, row2, row3, row4))

            end_time = time.perf_counter()

            # report time
            if verbose:
                print(f"lhs construction: {end_time - start_time} seconds.\n")
            
            return A
                
        # elif self.problem_id ==...

    def solve_system(self, verbose: bool=True) -> None:

        # Start the clock
        start_time = time.perf_counter()

        if self.__lhs is None:
            self.__lhs = self.compute_lhs(verbose, self.sparse)
        if self.__rhs is None:
            self.__rhs = self.compute_rhs(verbose)

        # Check if the shapes are (valid and) compatibile
        lhs_shape = self.__lhs.shape
        rhs_shape = self.__rhs.shape
        
        if lhs_shape[0] != lhs_shape[1]:
            raise ValueError(f"lhs matrix is not square: {lhs_shape[0]} by {lhs_shape[1]}.")
        if rhs_shape[0] != lhs_shape[0]:
            raise ValueError(f"lhs and rhs shapes are not compatibile: lhs of shape {lhs_shape[0]} by {lhs_shape[1]}, while rhs has length of {rhs_shape[0]}.")
        
        if self.sparse:
            self.__solution = linalg.spsolve(self.__lhs, self.__rhs)

            end_time = time.perf_counter()

            # report time
            if verbose:
                print(f"sparse solving time: {end_time - start_time} seconds.\n")
            
        else:
            self.__solution = np.linalg.solve(self.__lhs, self.__rhs)

            end_time = time.perf_counter()

            # report time
            if verbose:
                print(f"solving time: {end_time - start_time} seconds.\n")

    def set_lhs(self, lhs: np.ndarray) -> None:
        self.__lhs = lhs

    def set_rhs(self, rhs: np.ndarray) -> None:
        self.__rhs = rhs

    def get_dofs(self) -> tuple:

        """Accessor object method. Returns the degrees of freedom (DOFs) iff the system has been sucessfully solved.

        Raises:
        ------
            ValueError: Raised when the solution is not yet computed. Must self.solve_system(args) the solver first.

        Returns:
        -------
            (tuple): Returns the DOFs in the following order:
                        (w,     pi,                 sigma_kx,           sigma_ky,           w_T) with corresponding sizes:
                        (N ^ 3, N ^ 2 * (N + 1),    N ^ 2 * (N + 1),    N ^ 2 * (N + 1),    N ^ 2). Total length = 4 * N ^ 2 (N + 1).

        """ 
        
        if self.__solution is None:
            raise ValueError("The solution is not computeted yet. Compute it by self.solve_system(args).\n")
        
        N = self.get_metric().N
        solution = self.__solution

        w = solution[:N ** 3]
        pi = solution[N ** 3:2 * N ** 3 + N ** 2]
        sigma_kx = solution[2 * N ** 3 + N ** 2:3 * N ** 3 + 2 * N ** 2]
        sigma_ky = solution[3 * N ** 3 + 2 * N ** 2:4 * N ** 3 + 3 * N ** 2]
        w_T = solution[4 * N ** 3 + 3 * N ** 2: 4 * N ** 3 + 4 * N ** 2]

        # return the solution tuple in the order as below
        return (w, pi, sigma_kx, sigma_ky, w_T)
    
    def get_lhs(self) -> np.ndarray:
        return self.__lhs
    
    def get_rhs(self) -> np.ndarray:
        return self.__rhs
    
    def reconstruct_w(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:

        if isinstance(t, list):
            t = np.array(t)

        w = self.get_dofs()[0]
        metric = self.get_metric()
        rec = Reconstruction(metric, mesh_xieta_rec, t)

        if not isinstance(t, np.ndarray):
            return rec.volume_reconstruct(w, True, verbose=verbose)[0]
        return rec.volume_reconstruct(w, True, verbose=verbose)
    
    def reconstruct_z(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:

        if isinstance(t, list):
            t = np.array(t)

        w = self.get_dofs()[0]
        metric = self.get_metric()
        rec = Reconstruction(metric, mesh_xieta_rec, t)

        M_3 = metric.M_3(False)
        z = np.linalg.inv(M_3) @ w

        if not isinstance(t, np.ndarray):
            return rec.volume_reconstruct(z, False, verbose=verbose)[0]
        return rec.volume_reconstruct(z, False, verbose=verbose)
    
    def reconstruct_sigma_kx(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:

        if isinstance(t, list):
            t = np.array(t)

        sigma_x_dofs = self.get_dofs()[2]
        sigma_y_dofs = self.get_dofs()[3]

        metric = self.get_metric()
        rec: Reconstruction = Reconstruction(metric, mesh_xieta_rec, t)

        sigma_xi = rec.surface_reconstruct(sigma_x_dofs, 'x', verbose=verbose)
        sigma_eta = rec.surface_reconstruct(sigma_y_dofs, 'y', verbose=verbose)

        # Fix unwanted putting it in an array
        if not isinstance(t, np.ndarray):
            sigma_xi = sigma_xi[0]
            sigma_eta = sigma_eta[0]

        # Transform back
        sigma_kx = (sigma_xi * rec.X_XI_REC + sigma_eta * rec.X_ETA_REC) / rec.J_REC
        
        return sigma_kx
    
    def reconstruct_sigma_ky(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:

        if isinstance(t, list):
            t = np.array(t)

        sigma_x_dofs = self.get_dofs()[2]
        sigma_y_dofs = self.get_dofs()[3]

        metric = self.get_metric()
        rec: Reconstruction = Reconstruction(metric, mesh_xieta_rec, t)

        sigma_xi = rec.surface_reconstruct(sigma_x_dofs, 'x', verbose=verbose)
        sigma_eta = rec.surface_reconstruct(sigma_y_dofs, 'y', verbose=verbose)

        # Fix unwanted putting it in an array
        if not isinstance(t, np.ndarray):
            sigma_xi = sigma_xi[0]
            sigma_eta = sigma_eta[0]

        # Transform back
        sigma_ky = (sigma_xi * rec.Y_XI_REC + sigma_eta * rec.Y_ETA_REC) / rec.J_REC
        
        return sigma_ky
    
    def reconstruct_pi(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:

        if isinstance(t, list):
            t = np.array(t)

        pi_dofs = self.get_dofs()[1]

        metric = self.get_metric()
        rec: Reconstruction = Reconstruction(metric, mesh_xieta_rec, t)

        pi_tau = rec.surface_reconstruct(pi_dofs, 't', verbose=verbose)

        # Transform back
        pi = pi_tau / rec.J_REC * rec.T_TAU_REC
        
        # Fix unwanted putting it in an array
        if not isinstance(t, np.ndarray):
            pi = pi[0]


        return pi

    def reconstruct_div(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        
        if isinstance(t, list):
            t = np.array(t)

        metric = self.get_metric()
        utop = self.get_unit_topology()

        sigma_x_dofs = self.get_dofs()[2]
        sigma_y_dofs = self.get_dofs()[3]
        pi_dofs = self.get_dofs()[1]

        E_3_2_x = utop.E_3_2_x(False)
        E_3_2_y = utop.E_3_2_y(False)
        E_3_2_t = utop.E_3_2_t(False)

        flux_dofs = np.hstack((sigma_x_dofs, sigma_y_dofs, pi_dofs))
        E_3_2 = np.hstack((E_3_2_x, E_3_2_y, E_3_2_t))

        div = E_3_2 @ flux_dofs

        rec = Reconstruction(metric, mesh_xieta_rec, t)

        if not isinstance(t, np.ndarray):
            return rec.volume_reconstruct(div, False, verbose=verbose)[0]
        return rec.volume_reconstruct(div, False, verbose=verbose)

    def solve_plot_zw_h(self, dual: bool, \
                        N_x_rec: int, N_y_rec: int, \
                        t_slice: float, \
                        verbose: bool=True) -> None:
        
        if not isinstance(t_slice, float):
            raise ValueError("t_sclice must be of type float! It must be a 'slice' in time (single value).")
        
        if self.get_solution() is None:
            self.solve_system(verbose)

        print('\nPlotting...')

        d_map = self.get_metric().d_map
        t_map = self.get_metric().t_map

        mesh_xieta_rec = np.meshgrid(np.linspace(-1, 1, N_x_rec), np.linspace(-1, 1, N_y_rec))
        xi_rec_grid, eta_rec_grid = mesh_xieta_rec

        x_rec_grid = d_map.x(xi_rec_grid, eta_rec_grid)
        y_rec_grid = d_map.y(xi_rec_grid, eta_rec_grid)

        if dual:
            z_h = self.reconstruct_w(mesh_xieta_rec, t_slice)   # solve for w DOFs
        else:
            z_h = self.reconstruct_z(mesh_xieta_rec, t_slice)   # solve for z DOFs

        # Plotting setup
        fig_1 = Plotter3D(num_rows=1, num_cols=1)
        fig_1.plot_wireframe(0, x_rec_grid, y_rec_grid, z_h, \
                                cmap='magma', antialiased='True',
                                rstride=1, cstride=1, linewidth=0.5, edgecolor='black')
        
        fig_1.contour(0, x_rec_grid, y_rec_grid, z_h,\
                                zdir='z', offset=np.min(z_h) * 1.2, levels=np.linspace(-1, 1, 20),
                                linewidths=.8, colors='black')
        
        fig_1.set_limits(0, (np.min(x_rec_grid), np.max(x_rec_grid)), \
                            (np.min(y_rec_grid), np.max(y_rec_grid)), \
                            (np.min(z_h) * 1.2, np.max(z_h) * 1.2))

        # All done
        fig_1.set_style(0)
        fig_1.show_plot()

    def solve_plot_pi_h(self, \
                        N_x_rec: int, N_y_rec: int, \
                        t_slice: float, \
                        verbose: bool=True) -> None:
        
        if not isinstance(t_slice, float):
            raise ValueError("t_sclice must be of type float! It must be a 'slice' in time (single value).")
        
        if self.get_solution() is None:
            self.solve_system(verbose)

        print('\nPlotting...')

        d_map = self.get_metric().d_map
        t_map = self.get_metric().t_map

        mesh_xieta_rec = np.meshgrid(np.linspace(-1, 1, N_x_rec), np.linspace(-1, 1, N_y_rec))
        xi_rec_grid, eta_rec_grid = mesh_xieta_rec

        x_rec_grid = d_map.x(xi_rec_grid, eta_rec_grid)
        y_rec_grid = d_map.y(xi_rec_grid, eta_rec_grid)

        pi_h = self.reconstruct_pi(mesh_xieta_rec, t_slice)

        # Plotting setup
        fig_1 = Plotter3D(num_rows=1, num_cols=1)
        fig_1.plot_wireframe(0, x_rec_grid, y_rec_grid, pi_h, \
                                cmap='magma', antialiased='True',
                                rstride=1, cstride=1, linewidth=0.5, edgecolor='black')
        
        fig_1.contour(0, x_rec_grid, y_rec_grid, pi_h,\
                                zdir='z', offset=np.min(pi_h) * 1.2, levels=np.linspace(-1, 1, 20),
                                linewidths=.8, colors='black')
        
        fig_1.set_limits(0, (np.min(x_rec_grid), np.max(x_rec_grid)), \
                            (np.min(y_rec_grid), np.max(y_rec_grid)), \
                            (np.min(pi_h) * 1.2, np.max(pi_h) * 1.2))

        # All done
        fig_1.set_style(0)
        fig_1.show_plot() 

    def solve_animate_zw_h(self, dual: bool, \
                        N_x_rec: int, N_y_rec: int, \
                        t: Union[np.ndarray, list], \
                        verbose: bool=True, \
                        show: bool=True, save_name: Union[None, str]=None) -> None:
        
        if self.get_solution() is None:
            self.solve_system(verbose)

        print('\nAnimating...')

        d_map = self.get_metric().d_map

        mesh_xieta_rec = np.meshgrid(np.linspace(-1, 1, N_x_rec), np.linspace(-1, 1, N_y_rec))
        xi_rec_grid, eta_rec_grid = mesh_xieta_rec

        x_rec_grid = d_map.x(xi_rec_grid, eta_rec_grid)
        y_rec_grid = d_map.y(xi_rec_grid, eta_rec_grid)

        if dual:
            z_h = self.reconstruct_w(mesh_xieta_rec, t)   # solve for w DOFs
        else:
            z_h = self.reconstruct_z(mesh_xieta_rec, t)   # solve for z DOFs

        animator = Animator(x_rec_grid, y_rec_grid, z_h)

        if save_name is not None:
            animator.animation(show=show, save_name="res\\"+save_name)

    def solve_animate_pi_h(self, \
                        N_x_rec: int, N_y_rec: int, \
                        t: Union[np.ndarray, list], \
                        verbose: bool=True, \
                        show: bool=True, save_name: Union[None, str]=None) -> None:
        
        if self.get_solution() is None:
            self.solve_system(verbose)

        print('\nAnimating...')

        d_map = self.get_metric().d_map

        mesh_xieta_rec = np.meshgrid(np.linspace(-1, 1, N_x_rec), np.linspace(-1, 1, N_y_rec))
        xi_rec_grid, eta_rec_grid = mesh_xieta_rec

        x_rec_grid = d_map.x(xi_rec_grid, eta_rec_grid)
        y_rec_grid = d_map.y(xi_rec_grid, eta_rec_grid)

        pi_h = self.reconstruct_pi(mesh_xieta_rec, t)  # solve for z DOFs

        animator = Animator(x_rec_grid, y_rec_grid, pi_h)

        if save_name is not None:
            animator.animation(show=show, save_name="res\\"+save_name)
        
class MESolver(ValidProblemID, Reconstructible):

    """
    Multiple element solver class.

    Notes:
    ------
    - problem_id is the problem_id of the problem on the first cube (initial problem)
    - K is the  number of cubes
    """

    def __init__(self, problem_id: int, K: int, sparse: bool, verbose: bool=True, **kwargs) -> None:

        if problem_id not in Solver.valid_problem_id:
            raise ValueError(problem_id + " not valid. See valid ids: " + str(Solver.valid_problem_id))
        
        if not isinstance(K, int) and not isinstance(K, np.int32):
            raise TypeError(f"K must be an integer (type int), not type {type(K)}.")
        
        self.problem_id = problem_id
        self.K = K
        self.sparse = sparse
        self.verbose = verbose
        self.kwargs = kwargs

        self.initial_solver: Solver = Solver(problem_id, sparse, verbose, **kwargs)
        self.initial_problem: Problem = self.initial_solver.problem
        self.initial_metric: Metric = self.initial_solver.get_metric()

        # Adjust name
        self.initial_problem.description = self.initial_problem.description.replace("a SINGLE element", "MULTIPLE elements")
        # Add the K parameter
        self.initial_problem.K = K

        self.d_map: DomainMapping = self.initial_problem.d_map
        self.t_map: TimeMapping = self.initial_problem.t_map
        self.t0: float = self.t_map.t(-1.)
        self.tf: float = self.t_map.t(1.)
        self.dt: float = (self.tf - self.t0) / K
        self.time_divisions: np.ndarray = MESolver.generate_time_divisions(self.t0, self.tf, self.K)

        self.solver_chain: Union[Tuple[Solver], NoneType] = None

    @staticmethod
    def generate_time_divisions(t0: float, tf: float, K: float) -> np.ndarray:
        """
        Generate K time divisions (begin, end times) for a given time interval between t0 and tf.

        Parameters:
        t0 (float): The initial time of the interval.
        tf (float): The final time of the interval.
        K (float): The number of divisions.

        Returns:
        np.ndarray: An array of shape (K, 2) containing the initial and final time for each division.
        """

        # Create a numpy array with K rows and 2 columns
        time_divisions = np.zeros((K, 2))

        # Divide the time interval into K equal parts
        time_points = np.linspace(t0, tf, K+1)

        # Assign initial and final time for each division
        for i in range(K):
            time_divisions[i, 0] = time_points[i]  # Initial time
            time_divisions[i, 1] = time_points[i+1]  # Final time

        return time_divisions
    
    @staticmethod
    def split_time(general_t: np.ndarray, time_divisions: np.ndarray) -> Tuple[np.ndarray]:
        """
        Splits the given general_t array into multiple subarrays based on the provided time divisions.

        Args:
            general_t (np.ndarray): The array of general times.
            time_divisions (np.ndarray): The array of time divisions.

        Returns:
            Tuple[np.ndarray]: A tuple of subarrays, where each subarray contains the times within a specific division.

        """
        divided_times = []
        for start_time, end_time in time_divisions:
            # Filter times that are >= start_time and < end_time
            if end_time == time_divisions[-1][1]:  # If it's the last division
                divided_times.append(general_t[(general_t >= start_time) & (general_t <= end_time)])
            else:
                divided_times.append(general_t[(general_t >= start_time) & (general_t < end_time)])
        return tuple(divided_times)

    @staticmethod
    def compose_new_rhs_from_prev_soln(N: int, soln: np.ndarray) -> np.ndarray:
        
        # make unit topology
        utop = UnitTopology(N)

        I = soln[4]
        final_w = utop.N_2_0(verbose=False) @ I
        final_pi = soln[1][N * N ** 2:]

        new_rhs = np.hstack((np.zeros(N ** 3), final_w, np.zeros(2 * N ** 2 * (N + 1)), final_pi)).T

        return new_rhs

    def print_problem(self) -> None:
        print(self.initial_problem)

    def get_time_index(self, t: float) -> int:
        if not (self.t0 <= t <= self.tf):
            raise ValueError(f"Invalid time. Time {t} exceeds bounds [{self.t0}, {self.tf}].")
        
        for i, (start_time, end_time) in enumerate(self.time_divisions):
            # Check if the time is within the current division
            if start_time <= t <= end_time:
                return i

    # Public method calling the private one
    def solve(self, verbose: bool=True):
        self.solver_chain = self.__generate_solver_chain(verbose=verbose)

    # Private method generating the chain
    def __generate_solver_chain(self, verbose: bool) -> tuple:

        print("Generating the solver chain...")

        if verbose:
            print(f"k = {1} / {self.K}")

        # Only works for linear time mapping
        if not self.initial_metric.t_map.name == "linear":
            raise ValueError("ME so far only works with StandardTimeMapping linear.")
        
        # get initial solver, initial metric, order of the method
        i_solver = self.initial_solver
        i_metric = self.initial_metric
        N = i_metric.N

        # initialize the solver chain
        sc = np.empty(self.K, dtype=object)

        # Construct the first solver
        time_interval = self.time_divisions[0]

        curr_t_map = StandardTimeMapping("linear", t_begin=time_interval[0], t_end=time_interval[1])

        curr_metric = Metric(i_metric.N, i_metric.N_int_x, i_metric.N_int_y, i_metric.N_int_t, \
                                i_metric.c, i_metric.r, i_metric.d_map, curr_t_map)
        
        curr_solver = i_solver

        curr_solver.overwrite_metric(curr_metric)

        # Solve the first solver
        curr_solver.solve_system(False)

        # write in the first solver
        sc[0] = curr_solver

        # set current solution
        curr_soln = curr_solver.get_dofs()

        # begin looping throuh the remaining K-1 cubes
        for k in range(1, self.K):

            if verbose:
                print(f"k = {k + 1} / {self.K}")

            # get time interval
            time_interval = self.time_divisions[k]

            # make current time map
            curr_t_map = StandardTimeMapping("linear", t_begin=time_interval[0], t_end=time_interval[1])

            # construct current metric
            curr_metric = Metric(i_metric.N, i_metric.N_int_x, i_metric.N_int_y, i_metric.N_int_t, \
                                 i_metric.c, i_metric.r, i_metric.d_map, curr_t_map)
            
            # make the phantom solver
            curr_solver = Solver(0, sparse=self.sparse, verbose=self.verbose)
            
            # overwrite the metric in the phantom solver
            curr_solver.overwrite_metric(curr_metric)

            # compute lhs of current solver
            curr_solver.compute_lhs(self.verbose, self.sparse)

            # set rhs of current solver
            prev_soln = curr_soln
            curr_rhs = MESolver.compose_new_rhs_from_prev_soln(N, prev_soln)
            curr_solver.set_rhs(curr_rhs)

            # solve current system
            curr_solver.solve_system(False)

            # append the solved solver into the solver chain
            sc[k] = curr_solver

            # set current solution
            curr_soln = curr_solver.get_dofs()

            # delete the current solver
            del curr_solver

        return tuple(sc)

    # Define the ME reconstruction template (interface)
    def __reconstruct_any(self, name_of_reconstruction_method: str, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:

        # Check if the ME solver is solved
        if self.solver_chain is None:
            raise ValueError("The solution is not computeted yet. Compute it by self.solve().\n")

        # Convert to ndarray if float or list
        if isinstance(t, float):
            t = np.array([t])
        if isinstance(t, list):
            t = np.array(t)

        # Make the time split (tuple)
        time_split = MESolver.split_time(t, self.time_divisions)

        # Generate the reconstruction array
        rec_array = []

        # Reconstruct for each cube separately (if not empty)
        for k, t_per_cube in enumerate(time_split):
            # if empty, continue
            if len(t_per_cube) == 0:
                continue
            else:
                # current solver
                curr_solver = self.solver_chain[k]

                # Get the reconstruction method dynamically
                reconstruction_method = getattr(curr_solver, name_of_reconstruction_method)

                # Call the reconstruction method
                curr_rec = reconstruction_method(mesh_xieta_rec, t_per_cube, verbose=verbose)

                # append current reconstruction
                rec_array.append(curr_rec)

        # Concatenate
        rec_array = np.concatenate(rec_array, axis=0)

        return rec_array

    # Implement the parent class abstract methods by using the 'interface'
    def reconstruct_w(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        return self.__reconstruct_any('reconstruct_w', mesh_xieta_rec, t, verbose=verbose)
        
    def reconstruct_z(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        return self.__reconstruct_any('reconstruct_z', mesh_xieta_rec, t, verbose=verbose)
 
    def reconstruct_sigma_kx(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        return self.__reconstruct_any('reconstruct_sigma_kx', mesh_xieta_rec, t, verbose=verbose)
 
    def reconstruct_sigma_ky(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        return self.__reconstruct_any('reconstruct_sigma_ky', mesh_xieta_rec, t, verbose=verbose)

    def reconstruct_pi(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        return self.__reconstruct_any('reconstruct_pi', mesh_xieta_rec, t, verbose=verbose)
    
    def reconstruct_div(self, mesh_xieta_rec: np.ndarray, t: Union[float, np.ndarray, list], verbose=True) -> np.ndarray:
        return self.__reconstruct_any('reconstruct_div', mesh_xieta_rec, t, verbose=verbose)

    def get_condition_number(self) -> np.ndarray:
        if self.solver_chain is None:
            self.solve()
        
        condition_numbers = np.zeros(self.K)

        for k in range(self.K):
            curr_solver = self.solver_chain[k]
            condition_numbers[k] = curr_solver.get_condition_number()
        
        return condition_numbers
    
    def solve_plot_zw_h(self, dual: bool, \
                    N_x_rec: int, N_y_rec: int, \
                    t_slice: float, \
                    verbose: bool=True) -> None:
        
        # Check if the ME solver is solved
        if self.solver_chain is None:
            self.solve()

        # Check if t_slice is ONE value
        if not isinstance(t_slice, float):
            raise ValueError("t_sclice must be of type float! It must be a 'slice' in time (single value).")
        
        k = self.get_time_index(t_slice)
        
        # Choose solver
        solver = self.solver_chain[k]

        # Call plotting
        solver.solve_plot_zw_h(dual, N_x_rec, N_y_rec, t_slice, verbose=verbose)

    def solve_plot_pi_h(self, \
                    N_x_rec: int, N_y_rec: int, \
                    t_slice: float, \
                    verbose: bool=True) -> None:
        
        # Check if the ME solver is solved
        if self.solver_chain is None:
            self.solve()

        # Check if t_slice is ONE value
        if not isinstance(t_slice, float):
            raise ValueError("t_sclice must be of type float! It must be a 'slice' in time (single value).")
        
        k = self.get_time_index(t_slice)
        
        # Choose solver
        solver = self.solver_chain[k]

        # Call plotting
        solver.solve_plot_pi_h(N_x_rec, N_y_rec, t_slice, verbose=verbose)

    def solve_animate_zw_h(self, dual: bool, \
                        N_x_rec: int, N_y_rec: int, \
                        t: Union[np.ndarray, list], \
                        show: bool=True, save_name: Union[None, str]=None) -> None:
        
        if self.solver_chain is None:
            self.solve()

        print('\nAnimating...')

        d_map = self.d_map

        mesh_xieta_rec = np.meshgrid(np.linspace(-1, 1, N_x_rec), np.linspace(-1, 1, N_y_rec))
        xi_rec_grid, eta_rec_grid = mesh_xieta_rec

        x_rec_grid = d_map.x(xi_rec_grid, eta_rec_grid)
        y_rec_grid = d_map.y(xi_rec_grid, eta_rec_grid)

        if dual:
            z_h = self.reconstruct_w(mesh_xieta_rec, t)   # solve for w DOFs
        else:
            z_h = self.reconstruct_z(mesh_xieta_rec, t)   # solve for z DOFs

        animator = Animator(x_rec_grid, y_rec_grid, z_h)

        if save_name is not None:
            animator.animation(show=show, save_name="res\\"+save_name)

    def solve_animate_pi_h(self, \
                        N_x_rec: int, N_y_rec: int, \
                        t: Union[np.ndarray, list], \
                        show: bool=True, save_name: Union[None, str]=None) -> None:
        
        if self.solver_chain is None:
            self.solve()

        print('\nAnimating...')

        d_map = self.d_map

        mesh_xieta_rec = np.meshgrid(np.linspace(-1, 1, N_x_rec), np.linspace(-1, 1, N_y_rec))
        xi_rec_grid, eta_rec_grid = mesh_xieta_rec

        x_rec_grid = d_map.x(xi_rec_grid, eta_rec_grid)
        y_rec_grid = d_map.y(xi_rec_grid, eta_rec_grid)

        pi_h = self.reconstruct_pi(mesh_xieta_rec, t)  # solve for z DOFs

        animator = Animator(x_rec_grid, y_rec_grid, pi_h)

        if save_name is not None:
            animator.animation(show=show, save_name="res\\"+save_name)

if __name__ == "__main__":
    N0 = 3
    # s = Solver(problem_id=1, sparse=False, N=N0, N_int_x=N0, N_int_y=N0, N_int_t=N0+4, \
    #            t_map=StandardTimeMapping("linear", t_begin=0., t_end=2.), d_map = StandardDomainMapping("crazy_mesh", c=0.1), \
    #            verbose=False)
    # s.solve_system(False)
    # print(s.get_condition_number())
    # s.print_problem()

    # N_x_rec = 4
    # N_y_rec = 4

    # mesh_xi_eta_rec = np.meshgrid(np.linspace(-1, 1, N_x_rec), np.linspace(-1, 1, N_y_rec))
    
    # s.solve_system(False)
    # w_h = s.reconstruct_w(mesh_xi_eta_rec, [0., 1.])
    # # sigma_kx_h = s.reconstruct_sigma_kx(mesh_xi_eta_rec, [0., 1.])
    # # sigma_ky_h = s.reconstruct_sigma_ky(mesh_xi_eta_rec, [0., 1.])
    # # pi_h = s.reconstruct_pi(mesh_xi_eta_rec, 0.)

    # s.solve_animate_pi_h(50, 50, np.linspace(0., 2., 50), verbose=False, save_name='pi_soln_anim.gif')
    # s.solve_plot_pi_h(100, 100, 1., verbose=False)
    # s.solve_plot_zw_h(False, 100, 100, 1.)

    mes = MESolver(problem_id=1, K=6, sparse=True, N=N0, N_int_x=N0, N_int_y=N0, N_int_t=N0, t_map=StandardTimeMapping("linear", t_begin=0., t_end=2.), verbose=False)
    mes.print_problem()
    print(mes.get_condition_number())

    # mes.solve_plot_pi_h(100, 100, 0.97)
    # mes.solve_animate_zw_h(False, 100, 100, np.linspace(0., 2., 50), True, 'me_wh_test.gif')
