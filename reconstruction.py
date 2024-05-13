from assignment_functions import lobatto_quad, lagrange_basis, edge_basis
from components import Metric
from typing import Union, Literal
import numpy as np
import time

class Reconstruction(object):

    def __init__(self, metric: Metric, mesh_xieta_rec: np.ndarray, time_levels: Union[float, np.ndarray]) -> None:
        
        """Reconstruction object class. The constructor method.

        Args:
        ----
            metric (Metric): [Metric object]
            mesh_xieta_rec (np.ndarray): [xi-eta reconstruction mesh. (Constructed with np.meshgrid())]
            time_levels (Union[float, np.ndarray]): [The time level for which the reconstruction is to be done for. 
            Can be a scalar (one time level) or multiply time levels specified in a np.ndarray.]
        """

        # Convert the argument t to an array if only one value is given (not an array)
        if not isinstance(time_levels, np.ndarray):
            time_levels = np.array([time_levels])
        
    
        # Distribute attributes
        self.metric: Metric = metric
        self.mesh_xieta_rec = mesh_xieta_rec
        self.time_levels = time_levels

        # inverse transform required time levels
        tau_stripe = metric.t_map.inverse(time_levels)

        # Get the x and y component (grids) from np.meshgrid() (unpack)
        self.xi_rec_2d_grid, self.eta_rec_2d_grid = mesh_xieta_rec
        
        # Get reconstruction stripes (only distinct x, y, t values, not the entire grid)
        self.xi_stripe = self.xi_rec_2d_grid[0, :]
        self.eta_stripe = self.eta_rec_2d_grid[:, 0]
        self.tau_stripe = tau_stripe

        # For the purposes of evaluating transformation metric: Make the 3D grid (with all dimensions)
        self.eta_rec_grid, self.tau_rec_grid, self.xi_rec_grid = np.meshgrid(self.eta_stripe, self.tau_stripe, self.xi_stripe)

        # Bases evaluated at reconstruction nodes
        self.H_MATRIX_X_REC = lagrange_basis(self.metric.GL, self.xi_stripe)
        self.E_MATRIX_X_REC = edge_basis(self.metric.GL, self.xi_stripe)

        self.H_MATRIX_Y_REC = lagrange_basis(self.metric.GL, self.eta_stripe)
        self.E_MATRIX_Y_REC = edge_basis(self.metric.GL, self.eta_stripe)

        self.E_MATRIX_T_REC = edge_basis(self.metric.GL, tau_stripe)
        self.H_MATRIX_T_REC = lagrange_basis(self.metric.GL, tau_stripe)

        # Evaluate transformation metrics at reconstruction grid
        self.X_XI_REC = metric.d_map.x_xi(self.xi_rec_grid, self.eta_rec_grid)
        self.X_ETA_REC = metric.d_map.x_eta(self.xi_rec_grid, self.eta_rec_grid)
        self.Y_XI_REC = metric.d_map.y_xi(self.xi_rec_grid, self.eta_rec_grid)
        self.Y_ETA_REC = metric.d_map.y_eta(self.xi_rec_grid, self.eta_rec_grid)
        self.T_TAU_REC = metric.t_map.t_tau(self.tau_rec_grid)

        # The Jacobian evaluated at reconstruction grid
        self.J_REC = self.T_TAU_REC * (self.X_XI_REC * self.Y_ETA_REC - self.Y_XI_REC * self.X_ETA_REC)

        # Get the degree of the method
        self.N = metric.N

    def volume_reconstruct(self, solution: np.ndarray, dual: bool, verbose: bool=True) -> np.ndarray:

        # Start the clock
        start_time = time.perf_counter()
        
        N = self.N

        e_x = Metric._extend_repeat_matrix(self.E_MATRIX_X_REC, n=N * N, m=1)
        e_y = Metric._extend_repeat_matrix(self.E_MATRIX_Y_REC, n=N, m=N)
        e_t = Metric._extend_repeat_matrix(self.E_MATRIX_T_REC, n=1, m=N * N)
        primal_basis = Metric._tile_3_matrix(e_t, e_y, e_x)
        
        if not dual:
            R = np.einsum('i,ijkl->jkl', solution, primal_basis)

            end_time = time.perf_counter()

            # report time
            if verbose:
                print(f"primal volume reconstruction time: {end_time - start_time} seconds.\n")

            return R
        else:
            M_3_inv = np.linalg.inv(self.metric.M_3(False))
            dual_basis = np.einsum('ij,jklm->iklm', M_3_inv, primal_basis)

            R = np.einsum('i,ijkl->jkl', solution, dual_basis)

            end_time = time.perf_counter()

            # report time
            if verbose:
                print(f"dual volume reconstruction time: {end_time - start_time} seconds.\n")

            return R

    def surface_reconstruct(self, solution: np.ndarray, direction: Literal['x', 'y', 't'], verbose: bool=True) -> np.ndarray:
        
        # Direction is the normal direction to the surface that is to be reconstructed.
        # Solution must match the number of surfaces for a given mesh (only in the chosen direction)

        # Input check
        if (direction != 'x') and (direction != 'y') and (direction != 't'):
            raise RuntimeWarning(f"'{direction}' is not a valid reconstruction direction. Choose from: 'x', 'y', 't'.")

        # Start the clock
        start_time = time.perf_counter()

        N = self.N

        # Get the primal basis depending on the case
        if direction == 'x':
            
            h_x_x = Metric._extend_repeat_matrix(self.H_MATRIX_X_REC, m=1, n=N*N)
            e_x_y = Metric._extend_repeat_matrix(self.E_MATRIX_Y_REC, m=N+1, n=N)
            e_x_t = Metric._extend_repeat_matrix(self.E_MATRIX_T_REC, m=N*(N+1), n=1)

            primal_basis = Metric._tile_3_matrix(e_x_t, e_x_y, h_x_x)

        elif direction == 'y':

            e_y_x = Metric._extend_repeat_matrix(self.E_MATRIX_X_REC, m=1, n=N*(N+1))
            h_y_y = Metric._extend_repeat_matrix(self.H_MATRIX_Y_REC, m=N, n=N)
            e_y_t = Metric._extend_repeat_matrix(self.E_MATRIX_T_REC, m=N*(N+1), n=1)

            primal_basis = Metric._tile_3_matrix(e_y_t, h_y_y, e_y_x)

        else:
            
            e_t_x = Metric._extend_repeat_matrix(self.E_MATRIX_X_REC, m=1, n=N*(N+1))
            e_t_y = Metric._extend_repeat_matrix(self.E_MATRIX_Y_REC, m=N, n=N+1)
            h_t_t = Metric._extend_repeat_matrix(self.H_MATRIX_T_REC, m=N**2, n=1)

            primal_basis = Metric._tile_3_matrix(h_t_t, e_t_y, e_t_x)

        R = np.einsum('i,ijkl->jkl', solution, primal_basis)

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"primal surface '{direction}' reconstruction time: {end_time - start_time} seconds.\n")

        return R

        
