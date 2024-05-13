from assignment_functions import lobatto_quad, lagrange_basis, edge_basis
from typing import Callable
from mapping import TimeMapping, DomainMapping
import matplotlib.pyplot as plt
import numpy as np
import time

def cls():
    import os
    os.system("cls")

class UnitTopology(object):

    def __init__(self, N:int) -> None:

        # Distribute attributes
        self.N = N

    def __E_3_2_incorrect(self, verbose:bool=True)->np.ndarray:
        """
        Note: This function returns erroneous E32!
        """
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        # Auxiliary slip
        aux_slip = (-1, 1, np.zeros(N ** 2 * (N + 1) - 2),\
                    -1, np.zeros(N - 1), 1, np.zeros(N ** 3 + N **2 - N - 1),\
                    -1, np.zeros(N ** 2 -1), 1, np.zeros(N ** 3 - 1))

        aux_slip = np.hstack(aux_slip)
        
        # Prelocate memory
        E = np.zeros((N ** 3, 3 * N **2 * (N + 1)))

        # Roll the auxiliary slip
        for i in range(N ** 3):
            E[i, :] = np.roll(aux_slip, i)
        
        end_time = time.perf_counter()

        # Report time
        if verbose:
            print(f"E_3_2 incidence matrix: {end_time - start_time} seconds.\n")

        return E

    def E_3_2_x(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        aux_slip = (-1, 1, np.zeros(N ** 2 * (N + 1) - 2))
        aux_slip = np.hstack(aux_slip)

        E = np.zeros((N ** 3, N ** 2 * (N + 1)))

        # Roll the auxiliary slip
        for i in range(N ** 3):

            E[i, :] = aux_slip

            if (i + 1) % N == 0:
                aux_slip = np.roll(aux_slip, 2)                
            else:
                aux_slip = np.roll(aux_slip, 1)
        
        end_time = time.perf_counter()

        # Report time
        if verbose:
            print(f"E_3_2_x incidence matrix: {end_time - start_time} seconds.\n")

        return E
    
    def E_3_2_y(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        aux_slip = (-1, np.zeros(N - 1), 1, np.zeros(N ** 3 + N **2 - N - 1))
        aux_slip = np.hstack(aux_slip)

        E = np.zeros((N ** 3, N ** 2 * (N + 1)))

        # Roll the auxiliary slip
        for i in range(N ** 3):

            E[i, :] = aux_slip

            if (i + 1) % (N * N) == 0:
                aux_slip = np.roll(aux_slip, N + 1)                
            else:
                aux_slip = np.roll(aux_slip, 1)
        
        end_time = time.perf_counter()

        # Report time
        if verbose:
            print(f"E_3_2_y incidence matrix: {end_time - start_time} seconds.\n")

        return E
    
    def E_3_2_t(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        aux_slip = (-1, np.zeros(N ** 2 -1), 1, np.zeros(N ** 3 - 1))
        aux_slip = np.hstack(aux_slip)

        E = np.zeros((N ** 3, N ** 2 * (N + 1)))

        # Roll the auxiliary slip
        for i in range(N ** 3):

            E[i, :] = aux_slip

            aux_slip = np.roll(aux_slip, 1)
        
        end_time = time.perf_counter()

        # Report time
        if verbose:
            print(f"E_3_2_t incidence matrix: {end_time - start_time} seconds.\n")

        return E
    
    def N_2_0(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        E = np.vstack((-np.identity(N ** 2), np.zeros((N ** 3, N **2))))
        
        end_time = time.perf_counter()

        # Report time
        if verbose:
            print(f"N_2_0 inclusion matrix: {end_time - start_time} seconds.\n")

        return E
    
    def N_2_T(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        E = np.vstack((np.zeros((N ** 3, N **2)), np.identity(N ** 2)))
        
        end_time = time.perf_counter()

        # Report time
        if verbose:
            print(f"N_2_T inclusion matrix: {end_time - start_time} seconds.\n")

        return E
    
class Metric(object):

    def __init__(   self, N:int, N_int_x:int, N_int_y:int, N_int_t:int,\
                    c:Callable, r:Callable,\
                    d_map: DomainMapping, t_map: TimeMapping) -> None:

        # Distribute attributes
        self.N = N
        self.N_int_x = N_int_x
        self.N_int_y = N_int_y
        self.N_int_t = N_int_t
        self.c = c
        self.r = r
        self.d_map = d_map
        self.t_map = t_map

        # Gauss-Lobatto options
        self.GL, self.weights = lobatto_quad(self.N)
        self.GL_int_x, self.weights_int_x = lobatto_quad(self.N_int_x)
        self.GL_int_y, self.weights_int_y = lobatto_quad(self.N_int_y)
        self.GL_int_t, self.weights_int_t = lobatto_quad(self.N_int_t)

        # Unit integration grid
        self.XI_INT, self.ETA_INT, self.TAU_INT = np.meshgrid(self.GL_int_x, self.GL_int_y, self.GL_int_t)

        # NOTE; definition: physical integration grid = transformed(unit integration grid)
        # Transformed physical integration grid 
        self.X_INT = self.d_map.x(self.XI_INT, self.ETA_INT)
        self.Y_INT = self.d_map.y(self.XI_INT, self.ETA_INT)
        self.T_INT = self.t_map.t(self.TAU_INT)
        
        # Evaluate c and r on the physical integration grid
        self.C_EVAL = self.c(self.X_INT, self.Y_INT, self.T_INT)
        self.R_EVAL = self.r(self.X_INT, self.Y_INT, self.T_INT)
        
        # Bases evaluated at integration nodes
        self.H_MATRIX_X = lagrange_basis(self.GL, self.GL_int_x)
        self.E_MATRIX_X = edge_basis(self.GL, self.GL_int_x)

        self.H_MATRIX_Y = lagrange_basis(self.GL, self.GL_int_y)
        self.E_MATRIX_Y = edge_basis(self.GL, self.GL_int_y)

        self.H_MATRIX_T = lagrange_basis(self.GL, self.GL_int_t)
        self.E_MATRIX_T = edge_basis(self.GL, self.GL_int_t)

        # Evaluate transformation metrics at unit integration nodes
        self.X_XI_INT = self.d_map.x_xi(self.XI_INT, self.ETA_INT)
        self.X_ETA_INT = self.d_map.x_eta(self.XI_INT, self.ETA_INT)
        self.Y_XI_INT = self.d_map.y_xi(self.XI_INT, self.ETA_INT)
        self.Y_ETA_INT = self.d_map.y_eta(self.XI_INT, self.ETA_INT)
        self.T_TAU_INT = self.t_map.t_tau(self.TAU_INT)

        # The Jacobian evaluated at integration nodes
        self.J_INT = self.T_TAU_INT * (self.X_XI_INT * self.Y_ETA_INT - self.Y_XI_INT * self.X_ETA_INT)

    def M_0(self, verbose:bool=True)->np.ndarray:

        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        w_x = self.weights_int_x
        w_y = self.weights_int_y
        w_t = self.weights_int_t


        # Transformation
        # Begin by defining the 'Transforming Modifier' (TM): The additional term that needs to be integrated.
        # It usially consists of two pairs of two partials multiples and summed together (see notes), and the Jacobian inverse.
        # Specifically, here: TM = J
        TM = self.J_INT

        # Appropriate combinations
        h_x = self._extend_repeat_matrix(self.H_MATRIX_X, n=(N + 1) ** 2, m=1)
        h_y = self._extend_repeat_matrix(self.H_MATRIX_Y, n=N + 1, m=N + 1)
        h_t = self._extend_repeat_matrix(self.H_MATRIX_T, n=1, m=(N + 1) ** 2)

        # Tile the combinations ("normally span" three vectors containg vectors: e_x, e_y, h_t) and create
        # the T-tensor. The T-tensor is designed in such a way that it can be viewed as N^2 * (N+1)-long row vector,
        # whose entries are 3D tensors containg evaluated bases at the integration points ('integration cube').
        # When you outer-product T with T, that is T * T^(transpose), then the ij-th entry of constructed
        # such matrix is the entry of the mass matrix after integrating each ij-th 'integration cube' with appropriate weights.

        T = self._tile_3_matrix(h_x, h_y, h_t)

        M = np.einsum("iabc,jabc,bac,a,b,c->ij", T, T, TM, w_x, w_y, w_t, optimize="greedy")

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"M_0 mass matrix: {end_time - start_time} seconds.\n")

        return M

    def M_1_r(self, verbose:bool=True)->np.ndarray:

        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        w_x = self.weights_int_x
        w_y = self.weights_int_y
        w_t = self.weights_int_t

        R = self.R_EVAL

        # Transformation
        # Begin by defining the 'Transforming Modifier' (TM): The additional term that needs to be integrated.
        # It usially consists of two pairs of two partials multiples and summed together (see notes), and the Jacobian inverse.
        # Specifically, here: TM = (1 / J) * (dt/dtau) ^ 2
        TM = self.T_TAU_INT * self. T_TAU_INT / self.J_INT

        # Appropriate combinations
        e_x = self._extend_repeat_matrix(self.E_MATRIX_X, n=N * (N + 1), m=1)
        e_y = self._extend_repeat_matrix(self.E_MATRIX_Y, n=N + 1, m=N)
        h_t = self._extend_repeat_matrix(self.H_MATRIX_T, n=1, m=N ** 2)

        # Tile the combinations ("normally span" three vectors containg vectors: e_x, e_y, h_t) and create
        # the T-tensor. The T-tensor is designed in such a way that it can be viewed as N^2 * (N+1)-long row vector,
        # whose entries are 3D tensors containg evaluated bases at the integration points ('integration cube').
        # When you outer-product T with T, that is T * T^(transpose), then the ij-th entry of constructed
        # such matrix is the entry of the mass matrix after integrating each ij-th 'integration cube' with appropriate weights.

        T = self._tile_3_matrix(e_x, e_y, h_t)

        M = np.einsum("iabc,jabc,bac,a,b,c->ij", T, T, TM / R, w_x, w_y, w_t, optimize="greedy")

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"M_1_r mass matrix: {end_time - start_time} seconds.\n")

        return M

    def M_1_c_00(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        w_x = self.weights_int_x
        w_y = self.weights_int_y
        w_t = self.weights_int_t

        C = self.C_EVAL

        # Transformation
        # Begin by defining the 'Transforming Modifier' (TM): The additional term that needs to be integrated.
        # It usially consists of two pairs of two partials multiples and summed together (see notes), and the Jacobian inverse.
        # Specifically, here: TM = (1 / J) * (dx/dxi * dx/dxi + dy/dxi * dy/dxi)
        TM = (self.X_XI_INT * self.X_XI_INT + self.Y_XI_INT * self.Y_XI_INT) / self.J_INT

        # Appropriate combinations
        e_t = self._extend_repeat_matrix(self.E_MATRIX_T, n=1, m=N * (N + 1))
        h_x = self._extend_repeat_matrix(self.H_MATRIX_X, n=N ** 2, m=1)
        e_y = self._extend_repeat_matrix(self.E_MATRIX_Y, n=N, m=N + 1)   

        # Tile the combinations ("normally span" three vectors containg vectors: e_x, e_y, h_t) and create
        # the T-tensor. The T-tensor is designed in such a way that it can be viewed as N^2 * (N+1)-long row vector,
        # whose entries are 3D tensors containg evaluated bases at the integration points ('integration cube').
        # When you outer-product T with T, that is T * T^(transpose), then the ij-th entry of constructed
        # such matrix is the entry of the mass matrix after integrating each ij-th 'integration cube' with appropriate weights.

        T = self._tile_3_matrix(h_x, e_y, e_t)

        M = np.einsum("iabc,jabc,bac,a,b,c->ij", T, T, TM / C, w_x, w_y, w_t, optimize="greedy")

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"M_1_c_00 mass matrix: {end_time - start_time} seconds.\n")

        return M

    def M_1_c_01(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        w_x = self.weights_int_x
        w_y = self.weights_int_y
        w_t = self.weights_int_t

        C = self.C_EVAL

        # Transformation
        # Begin by defining the 'Transforming Modifier' (TM): The additional term that needs to be integrated.
        # It usially consists of two pairs of two partials multiples and summed together (see notes), and the Jacobian inverse.
        # Specifically, here: TM = (1 / J) * (dx/deta * dx/dxi + dy/deta * dy/dxi)
        TM = (self.X_ETA_INT * self. X_XI_INT + self.Y_ETA_INT * self.Y_XI_INT) / self.J_INT

        # Appropriate combinations
        e_x_1 = self._extend_repeat_matrix(self.E_MATRIX_X, n=N * (N + 1), m=1)
        h_y_1 = self._extend_repeat_matrix(self.H_MATRIX_Y, n=N, m=N)
        e_t_1 = self._extend_repeat_matrix(self.E_MATRIX_T, n=1, m=N * (N + 1))

        h_x_2 = self._extend_repeat_matrix(self.H_MATRIX_X, n=N ** 2, m=1)
        e_y_2 = self._extend_repeat_matrix(self.E_MATRIX_Y, n=N, m=N + 1)
        e_t_2 = self._extend_repeat_matrix(self.E_MATRIX_T, n=1, m=N * (N + 1))

        # Tile the combinations ("normally span" three vectors containg vectors: e_x, e_y, h_t) and create
        # the T_k-tensor. The T_k-tensor is designed in such a way that it can be viewed as N^2 * (N+1)-long row vector,
        # whose entries are 3D tensors containg evaluated bases at the integration points ('integration cube').
        # When you outer-product T1 with T2, that is T1 * T2, then the ij-th entry of constructed
        # such matrix is the entry of the mass matrix after integrating each ij-th 'integration cube' with appropriate weights.

        T1 = self._tile_3_matrix(e_x_1, h_y_1, e_t_1)
        T2 = self._tile_3_matrix(h_x_2, e_y_2, e_t_2)

        M = np.einsum("iabc,jabc,bac,a,b,c->ij", T2, T1, TM / C, w_x, w_y, w_t, optimize="greedy")

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"M_1_c_01 mass matrix: {end_time - start_time} seconds.\n")

        return M

    def M_1_c_11(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        w_x = self.weights_int_x
        w_y = self.weights_int_y
        w_t = self.weights_int_t

        C = self.C_EVAL
        # Transformation
        # Begin by defining the 'Transforming Modifier' (TM): The additional term that needs to be integrated.
        # It usially consists of two pairs of two partials multiples and summed together (see notes), and the Jacobian inverse.
        # Specifically, here: TM = (1 / J) * (dx/deta * dx/deta + dy/deta * dy/deta)
        TM = (self.X_ETA_INT * self.X_ETA_INT + self.Y_ETA_INT * self.Y_ETA_INT) / self.J_INT
        
        # Appropriate combinations
        e_t = self._extend_repeat_matrix(self.E_MATRIX_T, n=1, m=N * (N + 1))
        e_x = self._extend_repeat_matrix(self.E_MATRIX_X, n=N * (N + 1), m=1)
        h_y = self._extend_repeat_matrix(self.H_MATRIX_Y, n=N, m=N)   

        # Tile the combinations ("normally span" three vectors containg vectors: e_x, e_y, h_t) and create
        # the T-tensor. The T-tensor is designed in such a way that it can be viewed as N^2 * (N+1)-long row vector,
        # whose entries are 3D tensors containg evaluated bases at the integration points ('integration cube').
        # When you outer-product T with T, that is T * T^(transpose), then the ij-th entry of constructed
        # such matrix is the entry of the mass matrix after integrating each ij-th 'integration cube' with appropriate weights.

        T = self._tile_3_matrix(e_x, h_y, e_t)

        M = np.einsum("iabc,jabc,bac,a,b,c->ij", T, T, TM / C, w_x, w_y, w_t, optimize="greedy")

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"M_1_c_11 mass matrix: {end_time - start_time} seconds.\n")

        return M
    
    def M_3(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        w_x = self.weights_int_x
        w_y = self.weights_int_y
        w_t = self.weights_int_t


        # Appropriate combinations
        e_x = self._extend_repeat_matrix(self.E_MATRIX_X, n=N, m=N)
        e_y = self._extend_repeat_matrix(self.E_MATRIX_Y, n=N ** 2, m=1)
        e_t = self._extend_repeat_matrix(self.E_MATRIX_T, n=1, m=N ** 2)   

        # Tile the combinations ("normally span" three vectors containg vectors: e_x, e_y, e_t) and create
        # the T-tensor. The T-tensor is designed in such a way that it can be viewed as N^2 * (N+1)-long row vector,
        # whose entries are 3D tensors containg evaluated bases at the integration points ('integration cube').
        # When you outer-product T with T, that is T * T^(transpose), then the ij-th entry of constructed
        # such matrix is the entry of the mass matrix after integrating each ij-th 'integration cube' with appropriate weights.

        T = self._tile_3_matrix(e_x, e_y, e_t)

        M = np.einsum("iabc,jabc,a,b,c->ij", T, T, w_x, w_y, w_t, optimize="greedy")

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"M_3 mass matrix: {end_time - start_time} seconds.\n")

        return M

    def _depr_M_3_1D(self, verbose:bool=True)->np.ndarray:
        # Start the clock
        start_time = time.perf_counter()

        # Retrieve data
        N = self.N

        w_x = self.weights_int_x

        # Appropriate combinations
        e_x = self._extend_repeat_matrix(self.E_MATRIX_X, n=1, m=1)
        T = e_x

        # Tile the combinations ("normally span" three vectors containg vectors: e_x, e_y, e_t) and create
        # the T-tensor. The T-tensor is designed in such a way that it can be viewed as N^2 * (N+1)-long row vector,
        # whose entries are 3D tensors containg evaluated bases at the integration points ('integration cube').
        # When you outer-product T with T, that is T * T^(transpose), then the ij-th entry of constructed
        # such matrix is the entry of the mass matrix after integrating each ij-th 'integration cube' with appropriate weights.

        M = np.einsum("ia,ja,a->ij", T, T, w_x, optimize="greedy")

        end_time = time.perf_counter()

        # report time
        if verbose:
            print(f"M_3 mass matrix: {end_time - start_time} seconds.\n")

        return M

    # Private _-methods
    @staticmethod
    def _extend_repeat_matrix(T, n:int, m:int)->np.ndarray:
        """
            Args:
            T ([type]): tensor (2x2 matrix) to be repeated. Rows are the vectors to be repeated
            n (int): how many times to repeat the whole tensor (matrix)
            m (int): at each entry (matrix row=vector) how many times to repeat it
        """
        
        M1 = np.repeat(T, m, axis=0)
        M2 = np.tile(M1, (n, 1))

        return M2

    @staticmethod            
    def _tile_2_matrix(T1, T2)->np.ndarray:
        return np.einsum("ij,ik->ijk", T1, T2)
    
    @staticmethod
    def _tile_3_matrix(T1, T2, T3)->np.ndarray:
        return np.einsum("ij,ik,il->ijkl", T1, T2, T3)

    
if __name__ == "__main__":
    from mapping import StandardDomainMapping, StandardTimeMapping
    cls()

    c = lambda x, y, t: 0 * x + 0 * y + 0 * t + 1
    r = lambda x, y, t: 0 * x + 0 * y + 0 * t + 1
    d_map = StandardDomainMapping("crazy_mesh", c=0.1)
    t_map = StandardTimeMapping("linear", t_begin=0., t_end=3.)

    N0 = 2
    metric = Metric(N=N0, N_int_x=N0+5, N_int_y=N0, N_int_t=N0, c=c, r=r, d_map=d_map, t_map=t_map)
    utop = UnitTopology(N0)

    np.set_printoptions(linewidth=np.inf)
    print(utop.E_3_2_x(), utop.E_3_2_y(), utop.E_3_2_t())

    N_plt = 100
    xi = np.linspace(-1, 1, N_plt)

    M3 = metric._depr_M_3_1D()
    P_P3 = edge_basis(metric.GL, xi)
    P_D0 = (P_P3.T @ np.linalg.inv(M3)).T
    
    plt.figure()
    
    for i in range(N0):
        plt.plot(xi, P_D0[i, :], label=f"i={i}")

    plt.legend()
    plt.grid()
    plt.show()
