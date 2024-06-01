from functools import lru_cache
from math import atan, cos, exp, pi, sin, sqrt

import numpy as np
from mpmath import coth, csch

import np_uptake.model.cellular_uptake_rigid_particle as cellupt
from np_uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks
import tikzplotlib
from np_uptake.figures.utils import tikzplotlib_fix_ncols
from pathlib import Path
import seaborn as sns
class Fixed_Mechanical_Properties:
    """A class to represent the initial mechanical properties of the cell membrane and the particle.

    Attributes:
    ----------
    gamma_bar_0: float
        Adimensional linear adhesion energy between membrane and the particle, initial value
    sigma_bar: float
        Adimensional membrane tension, initial value

    Methods:
    -------
    None
    """

    def __init__(
        self,
        gamma_bar_0_list,
        sigma_bar_list,
    ):
        """Constructs all the necessary attributes for the Fixed_Mechanical_Properties object.

        Parameters:
        ----------
        gamma_bar_0: float
            Adimensional linear adhesion energy between membrane and the particle, initial value
        sigma_bar: float
            Adimensional membrane tension

        Returns:
        -------
        None
        """

        self.gamma_bar_0_list = gamma_bar_0_list
        self.sigma_bar_list = sigma_bar_list


class MechanicalProperties_Adaptation:
    """A class to represent the mechanical properties of the cell membrane and the particle.

    Attributes:
    ----------
    gamma_bar_0: float
        Adimensional linear adhesion energy between membrane and the particle, initial value
    gamma_bar_r: float
        Adimensional linear adhesion energy between membrane and the particle, ratio between
        final and initial value
    gamma_bar_fs: float
        Adimensional linear adhesion energy between membrane and the particle, inflexion point
    gamma_bar_lambda: float
        Adimensional linear adhesion energy between membrane and the particle, smoothness
    sigma_bar: float
        Adimensional membrane tension

    Methods:
    -------
    gamma_bar(self, f, wrapping):
        Returns the value of mechanical parameter gamma_bar for a given wrapping degree f
    plot_gamma_bar_variation(self, wrapping):
        Plots the evolution of both mechanical parameters with respect to wrapping degree f

    """

    def __init__(
        self,
        testcase,
        gamma_bar_r,
        gamma_bar_fs,
        gamma_bar_lambda,
        gamma_bar_0,
        sigma_bar,
    ):
        """Constructs all the necessary attributes for the mechanical properties object.

        Parameters:
        ----------
        testcase: string
            Identification name of the testcase. used to select the post treatment outfiles
        gamma_bar_0: float
            Adimensional linear adhesion energy between membrane and the particle, initial value
        gamma_bar_r: float
            Adimensional linear adhesion energy between membrane and the particle, ratio between
            final and initial value
        gamma_bar_fs: float
            Adimensional linear adhesion energy between membrane and the particle, inflexion point
        gamma_bar_lambda: float
            Adimensional linear adhesion energy between membrane and the particle, smoothness
        sigma_bar: float
            Adimensional membrane tension

        Returns:
        -------
        None
        """

        self.testcase = testcase
        self.gamma_bar_0 = gamma_bar_0
        self.gamma_bar_r = gamma_bar_r
        self.gamma_bar_fs = gamma_bar_fs
        self.gamma_bar_lambda = gamma_bar_lambda
        self.sigma_bar = sigma_bar

    @lru_cache(maxsize=128)
    def gamma_bar(self, f, wrapping):
        """Computes the adimensional value of the membrane-particle adhesion gamma_bar for a given
            wrapping degree f

        Parameters:
        ----------
        f: float
            Wrapping degree
        wrapping: class
            Wrapping class

        Returns:
        -------
        gamma_bar: float
            Adimensional membrane tension
        """

        if self.gamma_bar_r == 1:
            gamma_bar = self.gamma_bar_0
        else:
            gamma_bar_final = self.gamma_bar_r * self.gamma_bar_0
            x_list = np.linspace(-1, 1, len(wrapping.wrapping_list))
            gamma_bar = 0
            for i in range(len(wrapping.wrapping_list)):
                if wrapping.wrapping_list[i] == f:
                    x = x_list[i]
                    """
                    Sigmoid expression
                    """
                    if self.gamma_bar_lambda >= 0:
                        gamma_additional_term = self.gamma_bar_0
                    else:
                        gamma_additional_term = gamma_bar_final
                    gamma_bar += (abs(gamma_bar_final - self.gamma_bar_0)) * (
                        1 / (1 + exp(-self.gamma_bar_lambda * (x - self.gamma_bar_fs)))
                    ) + gamma_additional_term
            return gamma_bar
        return gamma_bar

    def gamma_bar_variation(self, wrapping):
        """Computes the values of the sigma_bar and gamma_bar for all the values of
            wrapping degree f

        Parameters:
        ----------
        wrapping: class
            Wrapping class

        Returns:
        -------
        sigma_bar_list: list
            Value of sigma_bar computed for all values of wrapping
        gamma_bar_list: list
            Value of gamma_bar computed for all values of wrapping
        """

        gamma_bar_list = [self.gamma_bar(f, wrapping) for f in wrapping.wrapping_list]
        return gamma_bar_list

    def plot_gamma_bar_variation(self, wrapping, createfigure, savefigure, fonts):
        """Plots the evolution of gamma_bar and sigma_bar with respect to the
            wrapping degree f

        Parameters:
        ----------
        wrapping: class
            Wrapping class

        Returns:
        -------
        None
        """

        fig = createfigure.rectangle_figure(pixels=360)
        ax = fig.gca()
        gamma_bar_list = self.gamma_bar_variation(wrapping)

        ax.plot(
            wrapping.wrapping_list,
            gamma_bar_list,
            "-k",
            label=r"$\overline{\gamma}_0 = $ ; "
            + str(self.gamma_bar_0)
            + r" ; $\overline{\gamma}_{D} = $ ; "
            + str(self.gamma_bar_fs)
            + r" ; $\overline{\gamma}_{A} = $ ; "
            + str(self.gamma_bar_r)
            + r" ; $\overline{\gamma}_{S} = $ ; "
            + str(self.gamma_bar_lambda),
        )
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.set_xticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )

        ax.set_yticks([gamma_bar_list[0], gamma_bar_list[-1]])
        ax.set_yticklabels(
            [np.round(gamma_bar_list[0], 2), np.round(gamma_bar_list[-1], 2)],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )

        ax.set_xlabel("f [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"$\overline{\gamma}$ [ - ] ", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.9)
        savefigure.save_as_png(fig, "adhesion_during_wrapping")
        tikzplotlib_fix_ncols(fig)
        current_path = Path.cwd()
        tikzplotlib.save(current_path/"adhesion_during_wrapping.tex")
        print('tkz ok')




    def plot_gamma_bar_variation_article(self, wrapping, createfigure, savefigure, fonts):
        """Plots the evolution of gamma_bar and sigma_bar with respect to the
            wrapping degree f

        Parameters:
        ----------
        wrapping: class
            Wrapping class

        Returns:
        -------
        None
        """

        fig = createfigure.rectangle_figure(pixels=360)
        ax = fig.gca()
        gamma_bar_list = self.gamma_bar_variation(wrapping)

        ax.plot(
            wrapping.wrapping_list,
            gamma_bar_list,
            "-k",
            label=r"$\overline{\gamma}_0 = $ ; "
            + str(self.gamma_bar_0)
            + r" ; $\overline{\gamma}_{D} = $ ; "
            + str(self.gamma_bar_fs)
            + r" ; $\overline{\gamma}_{A} = $ ; "
            + str(self.gamma_bar_r)
            + r" ; $\overline{\gamma}_{S} = $ ; "
            + str(self.gamma_bar_lambda),
        )
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.set_xticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )

        ax.set_yticks([gamma_bar_list[0], gamma_bar_list[-1]])
        ax.set_yticklabels(
            [np.round(gamma_bar_list[0], 2), np.round(gamma_bar_list[-1], 2)],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )

        ax.set_xlabel("f [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"$\overline{\gamma}$ [ - ] ", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.9)
        savefigure.save_as_png(fig, "adhesion_during_wrapping_article_fig5a")
        tikzplotlib_fix_ncols(fig)
        current_path = Path.cwd()
        tikzplotlib.save(current_path/"adhesion_during_wrapping_article_fig5a.tex")
        print('tkz ok')

class ParticleGeometry:
    """A class to define the geometry of the particle

    Attributes:
    ----------
    semi_minor_axis: float
        Semi-minor axis of the elliptic particle
    semi_major_axis: float
        Semi-major axis of the elliptic particle
    sampling_points_circle: int
        Number of points to describe a circular particle with radius 1
    f: float
        Wrapping degree
    theta: float
        Angle used as polar coordinate to define the ellipse, see figure * of readme

    Methods:
    -------
    define_particle_geometry_variables(self, f):
        Returns useful geometrical parameters to perform further calculations
    compute_r_coordinate(self, f, theta):
        Returns the value of the coordinate r given the position on the ellipse
    compute_z_coordinate(self, f, theta):
        Returns the value of the coordinate z given the position on the ellipse
    get_alpha_angle(self, f):
        Returns the value of the alpha angle (see figure *)
    compute_psi1_psi3_angles(self, f):
        Returns the curvature angles in the particle (see figure *)
    get_squared_dpsi_region3(self, f):
        Returns the values of dpsi3**2

    """

    def __init__(self, r_bar, particle_perimeter, sampling_points_circle):
        """Constructs all the necessary attributes for the particle object.

        Parameters:
        ----------
        r_bar: float
            Particle's aspect ratio
        particle_perimeter: float
            Particle's perimeter
        sampling_points_circle: int
            Number of points to describe a circular particle with radius semi_major axis.

        Returns:
        -------
        None
        """

        self.sampling_points_circle = sampling_points_circle
        self.r_bar = r_bar
        self.particle_perimeter = particle_perimeter

        # computes the semi-minor and semi-major axes lengths of the elliptic particle
        # using Ramanujan's formula [2]

        h = ((self.r_bar - 1) / (self.r_bar + 1)) ** 2
        self.semi_minor_axis = self.particle_perimeter / (pi * (1 + self.r_bar) * (1 + 3 * h / (10 + sqrt(4 - 3 * h))))
        self.semi_major_axis = self.semi_minor_axis * self.r_bar
        self.effective_radius = self.particle_perimeter / (2 * pi)

        # amount of points to sample the particle.
        self.sampling_points_ellipse = (
            self.sampling_points_circle * self.particle_perimeter / (2 * pi * self.semi_major_axis)
        )

    @lru_cache(maxsize=10)
    def define_particle_geometry_variables(self, f):
        """Defines all the necessary variables to describe the elliptic particle
        for a given wrapping degree f.

        Parameters:
        ----------
        f: float
            Wrapping degree (between 0 and 1)

        Returns:
        -------
        beta: float
            Trigonometric angle at intersection between regions 1, 2r and 3 (see figure *)
        beta_left: float
            Trigonometric angle at intersection between regions 1, 2l and 3 (see figure *)
        theta_list_region1: array
            Trigonomic angle theta into the region 1 (see figure *)
        theta_list_region3: array
            Trigonomic angle theta into the region 3 (see figure *)
        l1: float
            Arclength of the region 1 (see figure *)
        l3: float
            Arclength of the region 3 (see figure *)
        s_list_region1: array
            Sampling of the arclength of the region 1 (see figure *)
        s_list_region3: array
            Sampling of the arclength of the region 3 (see figure *)
        """

        beta = pi * f + 1.5 * pi
        beta_left = 3 * pi - beta
        n3 = int(f * self.sampling_points_ellipse)
        n1 = int((1 - f) * self.sampling_points_ellipse)
        theta_list_region1 = np.linspace(beta, 2 * pi + beta_left, n1)
        theta_list_region3 = np.linspace(beta_left, beta, n3)
        l1 = self.particle_perimeter * (1 - f)
        l3 = self.particle_perimeter * f
        s_list_region1 = np.linspace(0, l1, n1)
        s_list_region3 = np.linspace(0, l3, n3)
        return beta, beta_left, theta_list_region1, theta_list_region3, s_list_region3, l3, s_list_region1, l1

    @lru_cache(maxsize=128)
    def compute_r_coordinate(self, f, theta):
        """Computes the value of the coordinate r given the position on the ellipse,
            depicted by the angle theta, for a given wrapping degree f

        Parameters:
        ----------
        f: float
            Wrapping degree
        theta: float
            Trigonomic angle in the ellipse

        Returns:
        -------
        r_coordinate: float
            r coordinate (see figure *)
        """

        compute_x_coordinate = lambda t: self.semi_major_axis * cos(t)
        _, beta_left, _, _, _, _, _, _ = self.define_particle_geometry_variables(f)
        r_coordinate = compute_x_coordinate(theta) - compute_x_coordinate(beta_left)
        return r_coordinate

    @lru_cache(maxsize=128)
    def compute_z_coordinate(self, f, theta):
        """Computes the value of the coordinate z given the position on the ellipse,
            depicted by the angle theta, for a given wrapping degree f

        Parameters:
        ----------
        f: float
            Wrapping degree
        theta: float
            Trigonomic angle in the ellipse

        Returns:
        -------
        z_coordinate: float
            z coordinate (see figure *)
        """

        compute_y_coordinate = lambda t: self.semi_minor_axis * sin(t)
        _, beta_left, _, _, _, _, _, _ = self.define_particle_geometry_variables(f)
        z_coordinate = compute_y_coordinate(theta) - compute_y_coordinate(beta_left)
        return z_coordinate

    def compute_r_z_list(self, f):
        _, _, theta_list_region1, theta_list_region3, _, _, _, _ = self.define_particle_geometry_variables(f)
        r_list_region_1 = np.zeros_like(theta_list_region1)
        z_list_region_1 = np.zeros_like(theta_list_region1)
        r_list_region_3 = np.zeros_like(theta_list_region3)
        z_list_region_3 = np.zeros_like(theta_list_region3)   
        for i in range(len(theta_list_region1)):
            theta1 = theta_list_region1[i]
            r_list_region_1[i] = self.compute_r_coordinate(f, theta1)     
            z_list_region_1[i] = self.compute_z_coordinate(f, theta1)     
        for j in range(len(theta_list_region3)):
            theta3 = theta_list_region3[j]
            r_list_region_3[j] = self.compute_r_coordinate(f, theta3)     
            z_list_region_3[j] = self.compute_z_coordinate(f, theta3)  
        return  r_list_region_1, z_list_region_1, r_list_region_3, z_list_region_3

    @lru_cache(maxsize=10)
    def get_alpha_angle(self, f):
        """Computes the value of the alpha angle (see figure *), for a given wrapping degree f

        Parameters:
        ----------
        f: float
            Wrapping degree

        Returns:
        -------
        alpha: float
            Curvature angle at intersection between regions 1, 2l and 3 (see figure *)
        """

        psi_list_region1, _ = self.compute_psi1_psi3_angles(f)
        alpha = psi_list_region1[0]
        return alpha

    @lru_cache(maxsize=10)
    def compute_psi1_psi3_angles(self, f):
        """Computes the curvature angles in the particle (see figure *),
            for a given wrapping degree f

        Parameters:
        ----------
        f: float
            Wrapping degree

        Returns:
        -------
        psi_list_region1: list
            Psi angle in region 1 (see figure *)
        psi_list_region3: list
            Psi angle in region 3 (see figure *)
        """

        (
            beta,
            beta_left,
            theta_list_region1,
            theta_list_region3,
            s_list_region3,
            _,
            s_list_region1,
            _,
        ) = self.define_particle_geometry_variables(f)
        x_bl = self.semi_major_axis * cos(beta_left)

        def compute_psi_from_r_z(theta):
            """Computes the curvature angle given the position in the ellipse,
                depicted by theta (see figure *), using r and z coordinates

            Parameters:
            ----------
            theta: float
                Trigonomic angle in the ellipse

            Returns:
            -------
            delta: float
                Angle between the tangent to the particle and horizontal (see figure *)
            """

            r_elli = self.compute_r_coordinate(f, theta)
            z_elli = self.compute_z_coordinate(f, theta)

            def compute_tangent_to_ellipse_at_rtan_and_theta(r_tan):
                """Computes the position of the tangent to the particle (z with respect to r)

                Parameters:
                ----------
                r_tan: float
                    r coordinates where the tangent equation is evaluated

                Returns:
                -------
                z_tan: float
                    z coordinate of the tangent at r = r_tan
                """

                r_elli = self.compute_r_coordinate(f, theta)
                z_elli = self.compute_z_coordinate(f, theta)
                # managing possible singularities
                if r_elli == (-x_bl - self.semi_major_axis):
                    r_elli = r_elli + 0.01 * self.semi_major_axis
                elif r_elli == (-x_bl + self.semi_major_axis):
                    r_elli = r_elli - 0.01 * self.semi_major_axis
                slope1 = 1

                # depending on the side of the particle, the tangent's slope is positive or negative
                if theta > pi:
                    slope1 = -1
                dz = (
                    -self.semi_minor_axis
                    * (r_elli + x_bl)
                    / (self.semi_major_axis ** 2)
                    / sqrt(1 - ((r_elli + x_bl) / self.semi_major_axis) ** 2)
                )
                z_tan = z_elli + slope1 * dz * (r_tan - r_elli)
                return z_tan

            r1 = 1.5 * r_elli + 0.5
            z1 = compute_tangent_to_ellipse_at_rtan_and_theta(r1)
            delta = atan(abs((z1 - z_elli) / (r1 - r_elli)))
            return delta

        delta_list_region1 = [compute_psi_from_r_z(t) for t in theta_list_region1]
        delta_list_region3 = [compute_psi_from_r_z(t) for t in theta_list_region3]
        psi_list_region1 = np.zeros_like(s_list_region1)
        psi_list_region3 = np.zeros_like(s_list_region3)

        if f < 0.5:  # psi angle is defined differently depending on the position on the particle
            for i in range(len(s_list_region3)):
                theta = theta_list_region3[i]
                delta = delta_list_region3[i]
                if theta < 1.5 * pi:
                    psi = 2 * pi - delta
                elif theta == 1.5 * pi:
                    psi = 2 * pi
                elif theta <= beta:
                    psi = 2 * pi + delta
                psi_list_region3[i] = psi
            for i in range(len(s_list_region1)):
                theta = theta_list_region1[i]
                delta = delta_list_region1[i]
                if theta <= 2 * pi:
                    psi = delta
                elif theta <= 2 * pi + pi / 2:
                    psi = pi - delta
                elif theta <= 3 * pi:
                    psi = pi + delta
                else:
                    psi = 2 * pi - delta
                psi_list_region1[i] = psi
        else:
            for i in range(len(s_list_region3)):
                theta = theta_list_region3[i]
                delta = delta_list_region3[i]
                if theta < pi:
                    psi = pi + delta
                elif theta == pi:
                    psi = 1.5 * pi
                elif theta < 1.5 * pi:
                    psi = 2 * pi - delta
                elif theta == 1.5 * pi:
                    psi = 2 * pi
                elif theta < 2 * pi:
                    psi = 2 * pi + delta
                elif theta == 2 * pi:
                    psi = 2.5 * pi
                elif theta <= beta:
                    psi = 3 * pi - delta
                psi_list_region3[i] = psi

            for i in range(len(s_list_region1)):
                theta = theta_list_region1[i]
                delta = delta_list_region1[i]
                if theta < 2.5 * pi:
                    psi = pi - delta
                elif theta == 2.5 * pi:
                    theta = pi
                elif theta <= 2 * pi + beta_left:
                    psi = pi + delta
                psi_list_region1[i] = psi

        return psi_list_region1, psi_list_region3

    def get_squared_dpsi_region3(self, f):
        """Computes the values of dpsi3**2, necessary to evaluate the bending energy
            of the region 3, for a given wrapping degree f

        Parameters:
        ----------
        f: float
            Wrapping degree

        Returns:
        -------
        squared_dpsi_list_region3: list
            dpsi angle power 2 in region 3 (see figure *)
        """

        _, _, _, _, s_list_region3, _, _, _ = self.define_particle_geometry_variables(f)
        _, psi_list_region3 = self.compute_psi1_psi3_angles(f)
        ds = s_list_region3[1] - s_list_region3[0]
        # computes the derivative of psi in region 3 using finite differences method. The value of
        # ds was set after a convergence study.
        dpsi3_list_region3 = [
            (psi_list_region3[i + 1] - psi_list_region3[i]) / ds for i in range(0, len(psi_list_region3) - 1)
        ]
        squared_dpsi_list_region3 = [p ** 2 for p in dpsi3_list_region3]
        return squared_dpsi_list_region3


class MembraneGeometry:
    """A class to represent the membrane object.

    Attributes:
    ----------
    particle: class
        ParticleGeometry class
    mechanics: class
        MechanicalProperties class
    sampling_points_membrane: int
        Number of points to sample the regions 2r and 2l

    Methods:
    -------
    compute_r2r_r2l_z2r_z2l_from_analytic_expression(self, f, particle, mechanics):
        Returns r and z coordinates in the regions 2r and 2l
    """

    def __init__(self, particle, sampling_points_membrane):
        """Constructs all the necessary attributes for the membrane object.

        Parameters:
        ----------
        particle: class
            ParticleGeometry class
        sampling_points_membrane: int
            Number of points to sample the regions 2r and 2l

        Returns:
        -------
        None
        """

        self.sampling_points_membrane = sampling_points_membrane
        self.l2 = 20 * particle.effective_radius
        S2a = np.linspace(0, (self.l2 / 2), int((0.8 * self.sampling_points_membrane) + 1))
        S2b = np.linspace(1.2 * self.l2 / 2, self.l2, int(0.2 * self.sampling_points_membrane))
        self.S2 = np.concatenate((S2a, S2b), axis=None)

    @lru_cache(maxsize=10)
    def compute_r2r_r2l_z2r_z2l_from_analytic_expression(self, f, particle, mechanics, wrapping):
        """Computes the r and z coordinates to describe the regions 2r and 2l,
            for a given wrapping degree f

        Parameters:
        ----------
        f: float
            wrapping degree
        particle: class
            ParticleGeometry class
        mechanics: class
            MechanicalProperties class

        Returns:
        -------
        r2r: list
            r coordinate in the region 2r
        z2r: list
            z coordinate in the region 2l
        r2l: list
            r coordinate in the region 2r
        z2l: list
            z coordinate in the region 2l
        """

        _, _, _, theta_list_region3, _, _, _, _ = particle.define_particle_geometry_variables(f)
        alpha = particle.get_alpha_angle(f)
        r2r_0 = particle.compute_r_coordinate(f, theta_list_region3[-1])
        z2r_0 = particle.compute_z_coordinate(f, theta_list_region3[-1])
        r2r = np.zeros_like(self.S2)
        z2r = np.zeros_like(self.S2)
        sigma_bar = mechanics.sigma_bar

        for i in range(1, len(self.S2)):
            s = self.S2[i]
            r = (
                r2r_0
                + s
                - sqrt(2 / sigma_bar) * (1 - cos(alpha)) / (coth(s * sqrt(0.5 * sigma_bar)) + cos(alpha * 0.5))
            )
            z = z2r_0 + sqrt(8 / sigma_bar) * sin(0.5 * alpha) * (
                1 - (csch(s * sqrt(0.5 * sigma_bar))) / (coth(s * sqrt(0.5 * sigma_bar)) + cos(0.5 * alpha))
            )
            r2r[i] = r
            z2r[i] = z

        r2r[0] = r2r_0
        z2r[0] = z2r_0
        r2l = np.array([r2r[0] - r2r[s] for s in range(len(self.S2))])
        z2l = z2r
        return r2r, z2r, r2l, z2l


class Wrapping:
    """A class to represent the wrapping of the particle.

    Attributes:
    ----------
    wrapíng_list: list
        List of wrapping degrees at which the system is evaluated
    """

    def __init__(self, wrapping_list):
        """Constructs all the necessary attributes for the membrane object.

        Parameters:
        ----------
        wrapíng_list: list
            List of wrapping degrees at which the system is evaluated

        Returns:
        -------
        None
        """

        self.wrapping_list = wrapping_list

def plot_gamma_ratio_article(
    wrapping,
    createfigure,
    savefigure,
    fonts,
     xticks, xticklabels,
):
    testcase='0'
    gamma_bar_r_list = [1, 2, 3, 4]
    gamma_bar_fs, gamma_bar_lambda, gamma_bar_0, sigma_bar = 0, 50, 1, 2
    mechanoadaptation_list = [MechanicalProperties_Adaptation(testcase, gamma_bar_r, gamma_bar_fs, gamma_bar_lambda, gamma_bar_0, sigma_bar) for gamma_bar_r in gamma_bar_r_list]
    list_of_gamma_bar_r_vs_f = []
    for i in range(len(mechanoadaptation_list)):
        mechanoadaptation = mechanoadaptation_list[i]
        gamma_bar_r_vs_f = [mechanoadaptation.gamma_bar(f, wrapping) for f in wrapping.wrapping_list]
        list_of_gamma_bar_r_vs_f.append(gamma_bar_r_vs_f)
    fig = createfigure.square_figure_7(pixels=180)
    ax = fig.gca()
    palette = sns.color_palette("rocket", len(mechanoadaptation_list)-1)
    colors = ['k', palette[-2], palette[-1], palette[0]]
    kwargs = {"linewidth": 4}
    # r_bar_list_legend = ["1/5", "1/4", "1/3", "1/2", "1", "2", "3", "4", "5"]
    # linestyle = ["dotted", "dashed", "dashdot", "-", (0, (1, 1)), "-", "dashdot", "dashed", "dotted"]
    for i in range(len(list_of_gamma_bar_r_vs_f)):
        gamma_bar_r_vs_f = list_of_gamma_bar_r_vs_f[i]
        gamma_bar_r = gamma_bar_r_list[i]
        ax.plot(
            wrapping.wrapping_list,
            gamma_bar_r_vs_f,
            color=colors[-i],
            label=r"$\overline{\gamma}_A = $ " + str(gamma_bar_r),
            **kwargs,
        )
    # plt.legend()
    ax.set_xticks(xticks.energy_plots())
    ax.set_xticklabels(
        xticklabels.energy_plots(),
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(
        [1, 2, 3, 4],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_ylim((0.8, 4.2))
    ax.set_xlabel(r"$f $ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$\overline{\gamma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper left", framealpha=0.7)
    savefigure.save_as_png(
        fig,
        "params_sigmoid_gammaA_article_fig5a"
    )
    print('png ok')
    tikzplotlib_fix_ncols(fig)
    current_path = Path.cwd()
    tikzplotlib.save(current_path/"params_sigmoid_gammaA_article_fig5a.tex")
    print('tkz ok')

def plot_gamma_delay_article(
    wrapping,
    createfigure,
    savefigure,
    fonts,
    xticks,
    xticklabels,

):
    testcase = 'figure_params_sigmoid_gammaD'
    gamma_bar_fs_list = [-0.4, -0.2, 0, 0.2, 0.4]
    gamma_bar_r, gamma_bar_lambda, gamma_bar_0, sigma_bar = 2, 50, 1, 2
    mechanoadaptation_list = [MechanicalProperties_Adaptation(testcase, gamma_bar_r, gamma_bar_fs, gamma_bar_lambda, gamma_bar_0, sigma_bar) for gamma_bar_fs in gamma_bar_fs_list]
    list_of_gamma_bar_fs_vs_f = []
    for i in range(len(mechanoadaptation_list)):
        mechanoadaptation = mechanoadaptation_list[i]
        gamma_bar_fs_vs_f = [mechanoadaptation.gamma_bar(f, wrapping) for f in wrapping.wrapping_list]
        list_of_gamma_bar_fs_vs_f.append(gamma_bar_fs_vs_f)
    fig = createfigure.square_figure_7(pixels=180)
    ax = fig.gca()
    palette = sns.color_palette("Spectral", len(mechanoadaptation_list))
    # colors = [palette[k] for k in ]
    kwargs = {"linewidth": 4}
    # r_bar_list_legend = ["1/5", "1/4", "1/3", "1/2", "1", "2", "3", "4", "5"]
    # linestyle = ["dotted", "dashed", "dashdot", "-", (0, (1, 1)), "-", "dashdot", "dashed", "dotted"]
    for i in range(len(list_of_gamma_bar_fs_vs_f)):
        gamma_bar_fs_vs_f = list_of_gamma_bar_fs_vs_f[i]
        gamma_bar_fs = gamma_bar_fs_list[i]
        ax.plot(
            wrapping.wrapping_list,
            gamma_bar_fs_vs_f,
            color=palette[i],
            label=r"$\overline{\gamma}_D = $ " + str(gamma_bar_fs/2),
            **kwargs,
        )
    # plt.legend()
    ax.set_xticks(xticks.energy_plots())
    ax.set_xticklabels(
        xticklabels.energy_plots(),
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(
        [1, 2, 3, 4],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_ylim((0.95, 2.05))
    ax.set_xlabel(r"$f $ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$\overline{\gamma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper left", framealpha=0.7)
    savefigure.save_as_png(
        fig,
        "params_sigmoid_gammaD_article_fig5b",
    )
    print('png ok')
    tikzplotlib_fix_ncols(fig)
    current_path = Path.cwd()
    tikzplotlib.save(current_path/"params_sigmoid_gammaD_article_fig5b.tex")
    print('tkz ok')
    
def plot_gamma_slope_article(
    wrapping,
    createfigure,
    savefigure,
    fonts,
    xticks,
    xticklabels,
):
    testcase = 'figure_params_sigmoid_gammaS'
    gamma_bar_lambda_list = [0, 1, 2, 3, 4, 5, 10, 50, 100, 500]
    gamma_bar_r, gamma_bar_fs, gamma_bar_0, sigma_bar = 2, 0, 1, 2
    mechanoadaptation_list = [MechanicalProperties_Adaptation(testcase, gamma_bar_r, gamma_bar_fs, gamma_bar_lambda, gamma_bar_0, sigma_bar) for gamma_bar_lambda in gamma_bar_lambda_list]
    list_of_gamma_bar_lambda_vs_f = []
    
    for i in range(len(mechanoadaptation_list)):
        mechanoadaptation = mechanoadaptation_list[i]
        gamma_bar_lambda_vs_f = [mechanoadaptation.gamma_bar(f, wrapping) for f in wrapping.wrapping_list]
        list_of_gamma_bar_lambda_vs_f.append(gamma_bar_lambda_vs_f)
    
    fig = createfigure.square_figure_7(pixels=180)
    ax = fig.gca()
    palette = sns.color_palette("Spectral", len(mechanoadaptation_list))
    # colors = [palette[k] for k in ]
    kwargs = {"linewidth": 4}
    # r_bar_list_legend = ["1/5", "1/4", "1/3", "1/2", "1", "2", "3", "4", "5"]
    # linestyle = ["dotted", "dashed", "dashdot", "-", (0, (1, 1)), "-", "dashdot", "dashed", "dotted"]
    for i in range(len(list_of_gamma_bar_lambda_vs_f)):
        gamma_bar_lambda_vs_f = list_of_gamma_bar_lambda_vs_f[i]
        gamma_bar_lambda = gamma_bar_lambda_list[i]
        ax.plot(
            wrapping.wrapping_list,
            gamma_bar_lambda_vs_f,
            color=palette[i],
            label=r"$\overline{\gamma}_S = $ " + str(gamma_bar_lambda),
            **kwargs,
        )
    # plt.legend()
    ax.set_xticks(xticks.energy_plots())
    ax.set_xticklabels(
        xticklabels.energy_plots(),
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(
        [1, 2, 3, 4],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_ylim((0.95, 2.05))
    ax.set_xlabel(r"$f $ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$\overline{\gamma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper left", framealpha=0.7)
    savefigure.save_as_png(
        fig,
        "params_sigmoid_gammaS_article_fig5c"
    )

    print('png ok')
    tikzplotlib_fix_ncols(fig)
    current_path = Path.cwd()
    tikzplotlib.save(current_path/"params_sigmoid_gammaS_article_fig5c.tex")
    print('tkz ok')


if __name__ == "__main__":
    wrapping = Wrapping(wrapping_list=np.arange(0.03, 0.97, 0.003125))
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    args = cellupt.parse_arguments()
    xticks = XTicks()
    xticklabel = XTickLabels()

    mechanics = MechanicalProperties_Adaptation(
        testcase="test-classimplementation",
        gamma_bar_r=args.gamma_bar_r,
        gamma_bar_fs=args.gamma_bar_fs,
        gamma_bar_lambda=args.gamma_bar_lambda,
        gamma_bar_0=args.gamma_bar_0,
        sigma_bar=args.sigma_bar_0,
    )

    mechanics.plot_gamma_bar_variation(wrapping, createfigure, savefigure, fonts)
    plot_gamma_ratio_article(
    wrapping,
    createfigure,
    savefigure,
    fonts, xticks, xticklabel
)
    
    plot_gamma_ratio_article(
    wrapping,
    createfigure,
    savefigure,
    fonts, xticks, xticklabel
)
    
    plot_gamma_slope_article(
    wrapping,
    createfigure,
    savefigure,
    fonts, xticks, xticklabel
)
    
    plot_gamma_delay_article(
    wrapping,
    createfigure,
    savefigure,
    fonts, xticks, xticklabel
)
