# %%
## import some packages
import logging

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.visualization import Sphere


# %%
###########################################################################
#### Consider <n=3, p=1> Stiefel Manifold.
#### Dimension is 2, which can be represented as sphere.
###########################################################################
class StiefelSphere:
    """Class used to plot something in Stiefel shape space of 2D triangles,
    offering a 3D visualization of Kendall shape space of order
    (3,1), and its related objects.
    Attributes
    ----------
    
    References
    ----------

    """
    def __init__(self) -> None:
        self.points = []
        self.ax = None

        self.set_ax()

    def set_ax(self, ax=None):
        """Set axis."""
        if ax is None:
            ax = plt.subplot(111, projection="3d")

        ax_s = 0.5
        plt.setp(
            ax,
            xlim=(-ax_s, ax_s),
            ylim=(-ax_s, ax_s),
            zlim=(-ax_s, ax_s),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        self.ax = ax

    def set_view(self, elev=60.0, azim=0.0):
        """Set azimuth and elevation angle."""
        if self.ax is None:
            self.set_ax()

        self.elev, self.azim = gs.pi * elev / 180, gs.pi * azim / 180
        self.ax.view_init(elev, azim)

    def draw(self, n_theta=25, n_phi=13, scale=0.05, elev=60.0, azim=0.0):
        """Draw the sphere regularly sampled with corresponding triangles."""
        self.set_view(elev=elev, azim=azim)
        self.ax.set_axis_off()
        plt.tight_layout()

        coords_theta = gs.linspace(0.0, 2.0 * gs.pi, n_theta)
        coords_phi = gs.linspace(0.0, gs.pi, n_phi)

        coords_x = gs.to_numpy(0.5 * gs.outer(gs.sin(coords_phi), gs.cos(coords_theta)))
        coords_y = gs.to_numpy(0.5 * gs.outer(gs.sin(coords_phi), gs.sin(coords_theta)))
        coords_z = gs.to_numpy(
            0.5 * gs.outer(gs.cos(coords_phi), gs.ones_like(coords_theta))
        )

        self.ax.plot_surface(
            coords_x,
            coords_y,
            coords_z,
            rstride=1,
            cstride=1,
            color="grey",
            linewidth=0,
            alpha=0.1,
            zorder=-1,
        )
        self.ax.plot_wireframe(
            coords_x,
            coords_y,
            coords_z,
            linewidths=0.6,
            color="grey",
            alpha=0.6,
            zorder=-1,
        )

    def coordinates_transformation(self, points):
        return [x/2 for x in points]

    def add_points(self, points):
        if not isinstance(points, list):
            points = list(points)
        points = self.coordinates_transformation(points)
        self.points.extend(points)

    def draw_points(self, points=None, **scatter_kwargs):
        ax = self.ax
        if points is None:
            points = self.points
        points = [gs.autodiff.detach(point) for point in points]
        points = [gs.to_numpy(point) for point in points]
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        points_z = [point[2] for point in points]
        ax.scatter(points_x, points_y, points_z, **scatter_kwargs)

        for i_point, point in enumerate(points):
            if "label" in scatter_kwargs:
                if len(scatter_kwargs["label"]) == len(points):
                    ax.text(
                        point[0],
                        point[1],
                        point[2],
                        scatter_kwargs["label"][i_point],
                        size=10,
                        zorder=1,
                        color="k",
                    )

    def clear_points(self):
        """Clear the points to draw."""
        self.points = []

    def draw_mesh(self, point, **point_draw_kwargs):
        point = point.reshape(1,3)
        x0 = point[:,0]
        y0 = point[:,1]
        z0 = point[:,2]

        
        x_range = np.linspace(x0-0.4, x0+0.4, 10)
        y_range = np.linspace(y0-0.4, y0+0.4, 10)

        x_mesh, y_mesh = np.meshgrid(x_range, y_range)


        tangent_space = lambda x,y: z0 - (x0*(x-x0) + y0*(y-y0)) / z0

        z_mesh = tangent_space(x_mesh,y_mesh)        
        sphere = Sphere()
        ax = sphere.set_ax(ax=None)
        sphere.add_points(point)
        sphere.draw(ax, label='point', **point_draw_kwargs)
        ax.plot_surface(x_mesh, y_mesh, z_mesh,alpha=0.1)
        ax.legend()
        


# %%
###########################################################################
#### Consider <n=2, p=2 or 1> Stiefel Manifold.
#### Can be represented as circle.
###########################################################################
AX_SCALE = 2.5
S1 = Hypersphere(dim=1)
class StiefelCircle:
    """Class used to draw a circle."""

    def __init__(self, n_angles=100, points=None):
        angles = gs.linspace(0, 2 * gs.pi, n_angles)
        self.circle_x = gs.cos(angles)
        self.circle_y = gs.sin(angles)
        self.points = []
        self.ax = self.set_ax()
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), xlabel="X", ylabel="Y")
        return ax

    def add_points(self, points):
        if not gs.all(S1.belongs(points)):
            raise ValueError("Points do  not belong to the circle.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)
    
    def clear_points(self):
        self.points = []

    def draw(self, ax, **plot_kwargs):
        ax.plot(self.circle_x, self.circle_y, color="black")
        if self.points:
            self.draw_points(ax, **plot_kwargs)

    def draw_points(self, ax, points=None, **plot_kwargs):
        if points is None:
            points = self.points
        points = gs.array(points)
        ax.plot(points[:, 0], points[:, 1], marker="o", linestyle="None", **plot_kwargs)

    def draw_line_to_point(self, ax, point, line, **point_draw_kwargs):
        p_1 = point
        v_1 = line
        x1_range = np.linspace(p_1[0]-0.5, p_1[0]+0.5,1000)
        tan1_space = lambda x: (v_1[1] / v_1[0])*(x-p_1[0]) + p_1[1]
        ax.plot(x1_range, tan1_space(x1_range), '-', linewidth = 1, **point_draw_kwargs)

    def draw_curve(self, alpha=1, zorder=0, **kwargs):
        """Draw a curve on the Kendall disk."""
        vec_1 = self.points[0]
        vec_2 = self.points[1]
        angle_1 = np.degrees(np.arctan2(*vec_1.T[::-1])) % 360.0 * np.pi /180
        angle_2 = np.degrees(np.arctan2(*vec_2.T[::-1])) % 360.0 * np.pi /180
        min_angle = min(angle_1, angle_2)
        max_angle = max(angle_1, angle_2)
        if max_angle - min_angle > np.pi:
            set_angle = gs.linspace(max_angle, min_angle + 2 * np.pi, 1000)
        else:
            set_angle = gs.linspace(min_angle, max_angle, 1000)
        points_x = [np.cos(i) for i in set_angle]
        points_y = [np.sin(i) for i in set_angle]
        self.ax.plot(points_x, points_y, alpha=alpha, zorder=zorder, **kwargs)


    def draw_tangent_space(self, ax, base_point, **point_draw_kwargs):
        point = base_point.reshape(1,2)
        x0 = point[:,0]
        y0 = point[:,1]
        tangent_space = lambda x: np.linspace(-0.9,0.9,10) if y0 == 0 else (-x0/y0)*(x-x0) + y0
        x_range = x0*np.ones(10) if y0 == 0 else np.linspace(x0-0.9, x0+0.9, 10)
        ax.plot(x_range, tangent_space(x_range), '-', linewidth = 1, **point_draw_kwargs)
        ax.legend()

class Arrow2D:
    """An arrow in 3d, i.e. a point and a vector."""

    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def draw(self, ax, **quiver_kwargs):
        """Draw the arrow in 3D plot."""
        ax.quiver(
            self.point[0],
            self.point[1],
            self.vector[0],
            self.vector[1],
            **quiver_kwargs
        )
# %%
