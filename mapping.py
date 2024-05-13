from typing import Callable, Union
from point import Point
import matplotlib.pyplot as plt
import numpy as np

class TimeMapping(object):
    def __init__(self, t: Callable, t_tau: Callable, inverse: Callable) -> None:

        # Distribute attributes
        self.t: Callable = t
        self.t_tau: Callable = t_tau
        self.inverse: Callable = inverse

class StandardTimeMapping(TimeMapping):
    
    """
    Notes:
    1) All standard mappings names must begin with an underscore
    2) The return type of the standard mapping functions must be a tuple of Callables: (t, t_tau, inverse)
    3) In the lambdas always inlcude ALL parameters
    """

    def __init__(self, name: str, **kwargs) -> None:

        self.name = name
        self.kwargs = kwargs

        # create a list and a dictionary with all available names of mappings
        self.valid_names = [name[1:] for name in dir(StandardTimeMapping) if callable(getattr(StandardTimeMapping, name)) \
                            and not name.startswith("__") and not name == "plot"]

        self.__valid_names_dict = {name[1:]: value for name, value in StandardTimeMapping.__dict__.items() if callable(value) and not name.startswith("__")}

        # get the specified name as given in the initialization
        map = self.__valid_names_dict.get(name, None)

        # if the name is invalid print terminate
        if map is None:
            raise KeyError(name + " is not a valid name. See valid names: " + str(self.valid_names))
        
        # otherwise: get the mapping and initialize super()
        map = map(**kwargs)
        super().__init__(*map)        

    def __str__(self) -> str:
        out: str = self.name + "; "

        for key, value in self.kwargs.items():
            out += f"{key}={value}, "

        return out
    
    # Private standard mappings (templates, ready-to-go mappings)
    def _identity():
        t = lambda tau: tau
        t_tau = lambda tau: 1 + 0 * tau
        inverse = lambda t: t

        return t, t_tau, inverse
    
    def _linear(t_begin: float, t_end: float):
        t = lambda tau: 0.5 * (1 - tau) * t_begin + 0.5 * (1 + tau) * t_end
        t_tau = lambda tau: (t_end - t_begin) / 2 + 0 * tau
        inverse = lambda t: (2 * t - t_begin - t_end) / (t_end - t_begin)

        return t, t_tau, inverse
    
class DomainMapping(object):
    
    def __init__(self, x: Callable, y: Callable, x_xi: Union[Callable, None]=None, \
                 x_eta: Union[Callable, None]=None, y_xi: Union[Callable, None]=None,\
                      y_eta: Union[Callable, None]=None) -> None:
        
        # Distribute the attributes
        self.x = x
        self.y = y
        self.x_xi = x_xi
        self.x_eta = x_eta
        self.y_xi = y_xi
        self.y_eta = y_eta

    def plot(self, xiN: int=10, etaN: int=10, res: int=100, show: bool=False, figname: str=None, color=None, linewidth=None, save=False):
        
        if linewidth is None:
            linewidth = 1

        x = self.x
        y = self.y

        xi0 = np.linspace(-1, 1, xiN)
        eta0 = np.linspace(-1, 1, etaN)

        # begin figure
        plt.figure()

        # plot xi=xi0=const
        for xic in xi0:
            X, Y = [], []
            for etav in np.linspace(-1, 1, res):
                X.append(x(xic, etav))
                Y.append(y(xic, etav))
            
            if color is None:
                plt.plot(X, Y, color='g', linewidth=linewidth)
            else:
                plt.plot(X, Y, color=color, linewidth=linewidth)
        
        # plot eta=eta0=const
        for etac in eta0:
            X, Y = [], []
            for xiv in np.linspace(-1, 1, res):
                X.append(x(xiv, etac))
                Y.append(y(xiv, etac))
            
            if color is None:
                plt.plot(X, Y, color='b', linewidth=linewidth)
            else:
                plt.plot(X, Y, color=color, linewidth=linewidth)
        
        plt.axis('square')

        if save:
            if figname is None:
                from datetime import datetime, date
                ct = datetime.now().strftime("%Hh%Mm%Ss")
                cd = date.today()

                figname = str(self.T.__name__) + "_" + str(cd) + "_" + str(ct) + ".png"

            import os

            if not os.path.exists('domain_plots'):
                os.mkdir('domain_plots')

            plt.savefig('domain_plots'+'\\'+figname, dpi=500)
        
        if show:
            plt.show()

class StandardDomainMapping(DomainMapping):
    
    """
    Notes:
    1) All standard mappings names must begin with an underscore
    2) The return type of the standard mapping functions must be a tuple of Callables: (x, y, x_xi, x_eta, y_xi, y_eta)
    3) In the lambdas always inlcude ALL parameters
    """

    def __init__(self, name: str, **kwargs) -> None:

        self.kwargs = kwargs
        self.name = name

        # create a list and a dictionary with all available names of mappings
        self.valid_names = [name[1:] for name in dir(StandardDomainMapping) if callable(getattr(StandardDomainMapping, name)) \
                            and not name.startswith("__") and not name == "plot"]

        self.__valid_names_dict = {name[1:]: value for name, value in StandardDomainMapping.__dict__.items() if callable(value) and not name.startswith("__")}

        # get the specified name as given in the initialization
        map = self.__valid_names_dict.get(name, None)

        # if the name is invalid print terminate
        if map is None:
            raise KeyError(name + " is not a valid name. See valid names: " + str(self.valid_names))
        
        # otherwise: get the mapping and initialize super()
        map = map(**kwargs)
        super().__init__(*map)        

    def __str__(self) -> str:
        out: str = self.name + "; "

        for key, value in self.kwargs.items():
            out += f"{key}={value}, "

        return out

    # Private standard mappings (templates, ready-to-go mappings)
    def _identity():
        x = lambda xi, eta: xi + 0 * eta
        y = lambda xi, eta: 0 * xi + eta

        x_xi = lambda xi, eta: 1 + 0 * xi + 0 * eta
        x_eta = lambda xi, eta: 0 + 0 * xi + 0 * eta
        y_xi = lambda xi, eta: 0 + 0 * xi + 0 * eta
        y_eta = lambda xi, eta: 1 + 0 * xi + 0 * eta

        return x, y, x_xi, x_eta, y_xi, y_eta

    def _crazy_mesh(c: float):
        x = lambda xi, eta: xi + c * np.sin(np.pi * xi) * np.sin(np.pi * eta)
        y = lambda xi, eta: eta + c * np.sin(np.pi * xi) * np.sin(np.pi * eta)

        x_xi = lambda xi, eta: 1 + np.pi * c * np.cos(np.pi * xi) * np.sin(np.pi * eta)
        x_eta = lambda xi, eta: np.pi * c * np.sin(np.pi * xi) * np.cos(np.pi * eta)
        y_xi = lambda xi, eta: np.pi * c * np.cos(np.pi * xi) * np.sin(np.pi * eta)
        y_eta = lambda xi, eta: 1 + np.pi * c * np.sin(np.pi * xi) * np.cos(np.pi * eta)

        return x, y, x_xi, x_eta, y_xi, y_eta

    def _any_quad(P1: Point, P2: Point, P3: Point, P4: Point):

        x1, y1 = P1.x, P1.y
        x2, y2 = P2.x, P2.y
        x3, y3 = P3.x, P3.y
        x4, y4 = P4.x, P4.y

        X = np.array([x1, x2, x3, x4]).T
        Y = np.array([y1, y2, y3, y4]).T

        V = np.array([
        [1, -1, -1, 1],
        [1, 1, -1, -1],
        [1, 1, 1, 1],
        [1, -1, 1, -1]
        ])

        _ax, _bx, _cx, _dx = np.linalg.inv(V) @ X
        _ay, _by, _cy, _dy = np.linalg.inv(V) @ Y

        x = lambda xi, eta: _ax + _bx * xi + _cx * eta + _dx * xi * eta
        y = lambda xi, eta: _ay + _by * xi + _cy * eta + _dy * xi * eta
        x_xi = lambda xi, eta: _bx + _dx * eta
        x_eta = lambda xi, eta: _cx + _dx * xi
        y_xi = lambda xi, eta: _by + _dy * eta
        y_eta = lambda xi, eta: _cy + _dy * xi

        return x, y, x_xi, x_eta, y_xi, y_eta

    def _bent_tube(w: Union[Callable, None]=None, s: Union[Callable, None]=None, w_prime: Union[Callable, None]=None, s_prime: Union[Callable, None]=None):
        import numpy as np

        # Define how width w and separation s changes as a function of theta
        if w is None:
            w = lambda theta: 1 + 0.6 * np.cos(8 * theta)
            w_prime = lambda theta: -4.8 * np.sin(8 * theta)  # Derivative of w with respect to theta

        else:
            if w_prime is None:
                raise ValueError("The derivative function of w wrt. theta, i.e w_prime, must be passed if w is not the default function.")

        if s is None:
            s = lambda theta: 1 - 0.5 * (theta - np.pi / 6) * (theta - 2 * np.pi / 6)
            s_prime = lambda theta: -0.5 * ((theta - np.pi / 6) + (theta - 2 * np.pi / 6))  # Derivative of s with respect to theta
        else:
            if s_prime is None:
                raise ValueError("The derivative function of s wrt. theta, i.e s_prime, must be passed if s is not the default function.")

        # the relation between eta and theta is as follows
        # theta = np.pi / 4 * (eta + 1)

        x = lambda xi, eta: s(0) + w(0) - (s(np.pi / 4 * (eta + 1)) + 0.5 * w(np.pi / 4 * (eta + 1)) * (1 - xi)) * np.cos(np.pi / 4 * (eta + 1))
        y = lambda xi, eta: (s(np.pi / 4 * (eta + 1)) + 0.5 * w(np.pi / 4 * (eta + 1)) * (1 - xi)) * np.sin(np.pi / 4 * (eta + 1))

        x_xi = lambda xi, eta: 0.5 * w(np.pi / 4 * (eta + 1)) * np.cos(np.pi / 4 * (eta + 1)) + 0 * xi

        x_eta = lambda xi, eta: -s_prime(np.pi / 4 * (eta + 1)) * np.cos(np.pi / 4 * (eta + 1)) \
                - 0.5 * w_prime(np.pi / 4 * (eta + 1)) * (1 - xi) * np.cos(np.pi / 4 * (eta + 1)) \
                + 0.5 * w(np.pi / 4 * (eta + 1)) * np.sin(np.pi / 4 * (eta + 1))
        
        y_xi = lambda xi, eta: -0.5 * w(np.pi / 4 * (eta + 1)) * np.cos(np.pi / 4 * (eta + 1)) + 0 * xi

        y_eta = lambda xi, eta: s_prime(np.pi / 4 * (eta + 1)) * np.sin(np.pi / 4 * (eta + 1)) \
                + 0.5 * w_prime(np.pi / 4 * (eta + 1)) * (1 - xi) * np.sin(np.pi / 4 * (eta + 1)) \
                + 0.5 * w(np.pi / 4 * (eta + 1)) * np.cos(np.pi / 4 * (eta + 1))
        
        return x, y, x_xi, x_eta, y_xi, y_eta

    def _swirl(Smax: Union[float, None]=None, S00: Union[float, None]=None, r: Union[float, None]=None, c: Union[float, None]=None, a: Union[float, None]=None):

        if Smax is None:
            Smax: float = 0.15
        if S00 is None:
            S00: float = 0.024
        if r is None:
            r: float = 0.63
        if c is None:
            c: float = 0.9

        if a is None:
            a: float = 0.84

        # d is the domain shape function (max in the origin, 0 on the boundaries)
        d = lambda xi, eta: (1 - xi ** 2) * (1 - eta ** 2)
        # p is the 'particular' shape function (you set up the peaks and the radius with const 
        # predef. params.)
        p1 = lambda xi, eta: np.exp((abs(xi **2 + eta **2 - r **2)) * \
                                (np.log(S00 * (1 - c) * (1 - r ** 2)) / r ** 2))
        p2 = lambda xi, eta: np.sin(np.pi * (xi** 2 + eta ** 2) ** (- 0.5 * np.log(2) / np.log(a)))

        # s is the final shape function
        s1 = lambda xi, eta: Smax * min(1 / (1 - r ** 2) * 1 / (1 - c) * d(xi, eta) * p1(xi, eta), 1 + 0 * xi + 0 * eta)
        s2 = lambda xi, eta: 0.5 * d(xi, eta) * p2(xi, eta)

        x = lambda xi, eta: xi - s2(xi, eta) * eta / np.sqrt(xi **2 + eta ** 2)
        y = lambda xi, eta: eta + s2(xi, eta) * xi / np.sqrt(xi **2 + eta ** 2)

        return x, y, None, None, None, None

    def _polar():
        x = lambda xi, eta: xi * np.cos(eta)
        y = lambda xi, eta: xi * np.sin(eta)

        return x, y, None, None, None, None

if __name__ == "__main__":
    d_map = StandardDomainMapping("crazy_mesh", c=0.1)
    t_map = StandardTimeMapping("linear", t_begin=0, t_end=2)
    print(t_map)
