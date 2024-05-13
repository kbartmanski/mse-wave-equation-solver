'''Classes for solution plotting and visualisation

Author:
    Mark Nibbelke - November 2023
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as ANIM
from typing import Union


class Plotter3D:
    def __init__(self, num_rows=1, num_cols=1):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows),
                                           subplot_kw={'projection': '3d'})

        if num_rows == 1 and num_cols == 1:
            self.axes = np.array([self.axes])

    def add_subplot(self):
        self.num_cols += 1
        if self.num_cols > self.axes.shape[1]:
            self.num_rows += 1
            self.num_cols = 1
            self.axes = np.vstack([self.axes, plt.subplot(self.num_rows, self.num_cols, self.num_rows * self.num_cols)])
            self.fig.tight_layout()

        return self.axes[self.num_rows - 1, self.num_cols - 1]

    def plot_wireframe(self, ax, x, y, z, **kwargs):
        self.axes[ax].plot_wireframe(x, y, z, **kwargs)

    def contour(self, ax, x, y, z, **kwargs):
        self.axes[ax].contour(x, y, z, **kwargs)

    def set_limits(self, ax, xlims, ylims, zlims):
        self.axes[ax].set_xlim(xlims)
        self.axes[ax].set_ylim(ylims)
        self.axes[ax].set_zlim(zlims)

    def set_style(self, ax):
        self.axes[ax].view_init(elev=20, azim=-70)
        self.axes[ax].set_xlabel(r'$\xi$')
        self.axes[ax].set_ylabel(r'$\eta$')
        self.axes[ax].set_zlabel(r'$w^h(\xi, \eta, \tau)$')
        self.axes[ax].grid(False)
        self.axes[ax].xaxis.pane.set_edgecolor('black')
        self.axes[ax].yaxis.pane.set_edgecolor('black')
        self.axes[ax].zaxis.pane.set_edgecolor('black')
        self.axes[ax].xaxis.pane.set_alpha(1)
        self.axes[ax].yaxis.pane.set_alpha(1)
        self.axes[ax].zaxis.pane.set_alpha(1)
        self.axes[ax].xaxis.pane.fill = False
        self.axes[ax].yaxis.pane.fill = False
        self.axes[ax].zaxis.pane.fill = False
        self.axes[ax].set_frame_on(1)

    def set_subplot_title(self, ax, title: str):
        self.axes[ax].title.set_text(title)

    def set_fig_title(self, title):
        self.fig.suptitle(title)

    def show_plot(self):
        self.fig.tight_layout()
        plt.show()


class Animator:
    def __init__(self, x, y, rec):
        '''
        :param x: interpolation grid (x)
        :param y: interpolation grid (y)
        :param Rec: Reconstructed function at discrete times, on (x,y)
        '''
        self.x, self.y = x, y
        self.Data = rec
        self.Steps = self.Data.shape[0]
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.img = self.ax.imshow(self.Data[0], extent=[np.min(self.x), np.max(self.x),
                                                        np.max(self.y), np.min(self.y)], interpolation='none')

    def animation(self, save_name: Union[None, str]=None, show: bool=False):
        self.fig.gca().invert_yaxis()
        self.fig.colorbar(self.img, orientation='horizontal')
        anim = ANIM.FuncAnimation(self.fig, self._animate, self.Steps - 1, interval=200, repeat=False)
        writergif = ANIM.PillowWriter(fps=10)
        # noinspection PyTypeChecker
        if save_name is not None:
            anim.save(save_name, writer=writergif)
        if show:
            plt.show()

    def _animate(self, frame):
        '''This function updates
        the solution array'''
        CurrentSol = self.Data[frame]
        self.img.set_array(CurrentSol)
        self.img.set_clim(-1, 1)
        return self.img


class Plotter2D:
    def __init__(self, num_rows=1, num_cols=1):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))

        if num_rows == 1 and num_cols == 1:
            self.axes = np.array([self.axes])

    def contour(self, ax, x, y, z, **kwargs):
        self.axes[ax].contour(x, y, z, **kwargs)

    def set_limits(self, ax, xlims, ylims):
        self.axes[ax].set_xlim(xlims)
        self.axes[ax].set_ylim(ylims)

    def set_style(self, ax):
        self.axes[ax].view_init(elev=15, azim=-75)
        self.axes[ax].set_xlabel(r'$\xi$')
        self.axes[ax].set_ylabel(r'$\eta$')

    def set_subplot_title(self, ax, title: str):
        self.axes[ax].title.set_text(title)

    def set_fig_title(self, title):
        self.fig.suptitle(title)

    def show_plot(self):
        self.fig.tight_layout()
        plt.show()
