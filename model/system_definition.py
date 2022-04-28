from functools import lru_cache
from math import atan, cos, exp, pi, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
from mpmath import coth, csch


class Fixed_Mechanical_Properties:
    """
    A class to represent the mechanical properties of the cell membrane and the particle.

    Attributes:
        ----------
        gamma_bar_0: float
            adimensional linear adhesion energy between membrane and the particle, initial value
        sigma_bar_0: float
            adimensional membrane tension, initial value

    Methods:
        -------
        None
    """

    def __init__(
        self,
        gamma_bar_0_list,
        sigma_bar_list,
    ):
        """
        Constructs all the necessary attributes for the mechanical properties object.

        Parameters:
            ----------
            gamma_bar_0: float
                adimensional linear adhesion energy between membrane and the particle, initial value
            sigma_bar_0: float
                adimensional membrane tension, initial value
        Returns:
            -------
            None
        """
        self.gamma_bar_0_list = gamma_bar_0_list
        self.sigma_bar_list = sigma_bar_list


class MechanicalProperties_Adaptation:
    """
    A class to represent the mechanical properties of the cell membrane and the particle.

    Attributes:
        ----------
        gamma_bar_0: float
            adimensional linear adhesion energy between membrane and the particle, initial value
        gamma_bar_r: float
            adimensional linear adhesion energy between membrane and the particle, ratio between
            final and initial value
        gamma_bar_fs: float
            adimensional linear adhesion energy between membrane and the particle, inflexion point
        gamma_bar_lambda: float
            adimensional linear adhesion energy between membrane and the particle, smoothness
        sigma_bar_0: float
            adimensional membrane tension, initial value
        sigma_bar_r: float
            adimensional membrane tension, ratio between final and initial value
        sigma_bar_fs: float
            adimensional membrane tension, inflexion point
        sigma_bar_lambda: float
            adimensional membrane tension, smoothness

    Methods:
        -------
        sigma_bar(self, f, wrapping):
            Returns the value of mechanical parameter sigma_bar for a given wrapping degree f
        gamma_bar(self, f, wrapping):
            Returns the value of mechanical parameter gamma_bar for a given wrapping degree f
        plot_mechanical_parameters(self, wrapping):
            Plots the evolution of both mechanical parameters with respect to wrapping degree f

    """

    def __init__(
        self,
        testcase,
        gamma_bar_r,
        gamma_bar_fs,
        gamma_bar_lambda,
        gamma_bar_0,
        sigma_bar_0,
    ):
        """
        Constructs all the necessary attributes for the mechanical properties object.

        Parameters:
            ----------
            testcase: string
                identification name of the testcase. used to select the post treasment outfiles
            gamma_bar_0: float
                adimensional linear adhesion energy between membrane and the particle, initial value
            gamma_bar_r: float
                adimensional linear adhesion energy between membrane and the particle, ratio between
                final and initial value
            gamma_bar_fs: float
                adimensional linear adhesion energy between membrane and the particle, inflexion point
            gamma_bar_lambda: float
                adimensional linear adhesion energy between membrane and the particle, smoothness
            sigma_bar_0: float
                adimensional membrane tension, initial value
            sigma_bar_r: float
                adimensional membrane tension, ratio between final and initial value
            sigma_bar_fs: float
                adimensional membrane tension, inflexion point
            sigma_bar_lambda: float
                adimensional membrane tension, smoothness

        Returns:
            -------
            None
        """
        self.testcase = testcase
        self.gamma_bar_0 = gamma_bar_0
        self.gamma_bar_r = gamma_bar_r
        self.gamma_bar_fs = gamma_bar_fs
        self.gamma_bar_lambda = gamma_bar_lambda
        self.sigma_bar_0 = sigma_bar_0
        self.sigma_bar_r = sigma_bar_r
        self.sigma_bar_fs = sigma_bar_fs
        self.sigma_bar_lambda = sigma_bar_lambda

    @lru_cache(maxsize=128)
    def sigma_bar(self, f, wrapping):
        """
        Computes the adimensional value of the membrane tension sigma_bar for a given
            wrapping degree f

        Parameters:
            ----------
            f: float
                wrapping degree
            wrapping: class
                Wrapping class

        Returns:
            -------
            sigma_bar: float
                adimensional membrane tension
        """
        if self.sigma_bar_r == 1:
            sigma_bar = self.sigma_bar_0
        else:
            sigma_bar_final = self.sigma_bar_r * self.sigma_bar_0
            x_list = np.linspace(-1, 1, len(wrapping.wrapping_list))
            sigma_bar = 0
            for i in range(len(wrapping.wrapping_list)):
                if wrapping.wrapping_list[i] == f:
                    x = x_list[i]
                    """
                    Sigmoid expression
                    """
                    if self.sigma_bar_lambda > 0:
                        sigma_additional_term = self.sigma_bar_0
                    else:
                        sigma_additional_term = sigma_bar_final
                    sigma_bar += (abs(sigma_bar_final - self.sigma_bar_0)) * (
                        1 / (1 + exp(-self.sigma_bar_lambda * (x - self.sigma_bar_fs)))
                    ) + sigma_additional_term
        return sigma_bar

    @lru_cache(maxsize=128)
    def gamma_bar(self, f, wrapping):
        """
        Computes the adimensional value of the membrane-particle adhesion sigma_bar for a given
            wrapping degree f

        Parameters:
            ----------
            f: float
                wrapping degree
            wrapping: class
                Wrapping class

        Returns:
            -------
            gamma_bar: float
                adimensional membrane tension
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
                    if self.gamma_bar_lambda > 0:
                        gamma_additional_term = self.gamma_bar_0
                    else:
                        gamma_additional_term = gamma_bar_final
                    gamma_bar += (abs(gamma_bar_final - self.gamma_bar_0)) * (
                        1 / (1 + exp(-self.gamma_bar_lambda * (x - self.gamma_bar_fs)))
                    ) + gamma_additional_term
            return gamma_bar
        return gamma_bar

    def mechanical_parameters_evolution(self, wrapping):
        """
        Computes the values of the sigma_bar and gamma_bar for all the values of
            wrapping degree f

        Parameters:
            ----------
            wrapping: class
                Wrapping class

        Returns:
            -------
            sigma_bar_list: list
                value of sigma_bar computed for all values of wrapping
            gamma_bar_list: list
                value of gamma_bar computed for all values of wrapping
        """
        sigma_bar_list = [self.sigma_bar(f, wrapping) for f in wrapping.wrapping_list]
        gamma_bar_list = [self.gamma_bar(f, wrapping) for f in wrapping.wrapping_list]
        return sigma_bar_list, gamma_bar_list

    def plot_mechanical_parameters(self, wrapping):
        """
        Plots the evolution of gamma_bar and sigma_bar with respect to the
            wrapping degree f

        Parameters:
            ----------
            wrapping: class
                Wrapping class

        Returns:
            -------
            None
        """
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        sub1 = axs[0]  # plt.subplot(2, 1, 1)
        sub2 = axs[1]  # plt.subplot(2, 1, 2)
        sigma_bar_list, gamma_bar_list = self.mechanical_parameters_evolution(wrapping)
        sub1.plot(
            wrapping.wrapping_list,
            sigma_bar_list,
            "-k",
            label=r"$\overline{\sigma}_0 = $ ; "
            + str(self.sigma_bar_0)
            + r"$\overline{\sigma}_{fs} = $ ; "
            + str(self.sigma_bar_fs)
            + r"$\overline{\sigma}_{r} = $ ; "
            + str(self.sigma_bar_r)
            + r"$\overline{\sigma}_{\lambda} = $ ; "
            + str(self.sigma_bar_lambda),
        )
        sub2.plot(
            wrapping.wrapping_list,
            gamma_bar_list,
            "-k",
            label=r"$\overline{\gamma}_0 = $ ; "
            + str(self.gamma_bar_0)
            + r"$\overline{\gamma}_{fs} = $ ; "
            + str(self.gamma_bar_fs)
            + r"$\overline{\gamma}_{r} = $ ; "
            + str(self.gamma_bar_r)
            + r"$\overline{\gamma}_{\lambda} = $ ; "
            + str(self.gamma_bar_lambda),
        )
        sub1.set_xlabel("wrapping degree f [ - ]")
        sub1.set_ylabel(r"$\overline{\sigma}$")
        sub2.set_xlabel("wrapping degree f [ - ]")
        sub2.set_ylabel(r"$\overline{\gamma}$")
        sub1.legend()
        sub2.legend()


class ParticleGeometry:
    """
    A class to represent the particle.

    Attributes:
        ----------
        semi_minor_axis: float
            semi-minor axis of the elliptic particle
        semi_major_axis: float
            semi-major axis of the elliptic particle
        sampling_points_circle: int
            number of points to describe a circular particle with radius 1
        f: float
            wrapping degree
        theta: float
            angle used as polar coordinate to define the ellipse, see figure * of readme

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
        """
        Constructs all the necessary attributes for the particle object.

        Parameters:
            ----------
            r_bar: float
                particle's aspect ratio
            particle_perimeter: float
                particle's perimeter
            sampling_points_circle: int
                number of points to describe a circular particle with radius semi_major axis.

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
        """
        Defines all the necessary variables to describe the elliptic particle
        for a given wrapping degree f.

        Parameters:
            ----------
            f: float
                wrapping degree (between 0 and 1)

        Returns:
            -------
            beta: float
                trigonometric angle at intersection between regions 1, 2r and 3 (see figure *)
            beta_left: float
                trigonometric angle at intersection between regions 1, 2l and 3 (see figure *)
            theta_list_region1: array
                trigonomic angle theta into the region 1 (see figure *)
            theta_list_region3: array
                trigonomic angle theta into the region 3 (see figure *)
            l1: float
                arclength of the region 1 (see figure *)
            l3: float
                arclength of the region 3 (see figure *)
            s_list_region1: array
                sampling of the arclength of the region 1 (see figure *)
            s_list_region3: array
                sampling of the arclength of the region 3 (see figure *)

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
        """
        Computes the value of the coordinate r given the position on the ellipse,
            depicted by the angle theta, for a given wrapping degree f

        Parameters:
            ----------
            f: float
                wrapping degree
            theta: float
                trigonomic angle in the ellipse

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
        """
        Computes the value of the coordinate z given the position on the ellipse,
            depicted by the angle theta, for a given wrapping degree f

        Parameters:
            ----------
            f: float
                wrapping degree
            theta: float
                trigonomic angle in the ellipse

        Returns:
            -------
            z_coordinate: float
                z coordinate (see figure *)
        """
        compute_y_coordinate = lambda t: self.semi_minor_axis * sin(t)
        _, beta_left, _, _, _, _, _, _ = self.define_particle_geometry_variables(f)
        z_coordinate = compute_y_coordinate(theta) - compute_y_coordinate(beta_left)
        return z_coordinate

    @lru_cache(maxsize=10)
    def get_alpha_angle(self, f):
        """
        Computes the value of the alpha angle (see figure *), for a given wrapping degree f

        Parameters:
            ----------
            f: float
                wrapping degree

        Returns:
            -------
            alpha: float
                curvature angle at intersection between regions 1, 2l and 3 (see figure *)

        """
        psi_list_region1, _ = self.compute_psi1_psi3_angles(f)
        alpha = psi_list_region1[0]
        return alpha

    @lru_cache(maxsize=10)
    def compute_psi1_psi3_angles(self, f):
        """
        Computes the curvature angles in the particle (see figure *),
            for a given wrapping degree f

        Parameters:
            ----------
            f: float
                wrapping degree

        Returns:
            -------
            psi_list_region1: list
                psi angle in region 1 (see figure *)
            psi_list_region3: list
                psi angle in region 3 (see figure *)

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
            """
            Computes the curvature angle given the position in the ellipse,
                depicted by theta (see figure *), using r and z coordinates

            Parameters:
                ----------
                theta: float
                    trigonomic angle in the ellipse

            Returns:
                -------
                delta: float
                    angle between the tangent to the particle and horizontal (see figure *)

            """
            r_elli = self.compute_r_coordinate(f, theta)
            z_elli = self.compute_z_coordinate(f, theta)

            def compute_tangent_to_ellipse_at_rtan_and_theta(r_tan):
                """
                Computes the position of the tangent to the particle (z with respect to r)

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
                """
                Returns:
                    Eq of the tangent to the ellipse at theta = beta
                """
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
        """
        Computes the values of dpsi3**2, necessary to evaluate the bending energy
            of the region 3, for a given wrapping degree f

        Parameters:
            ----------
            f: float
                wrapping degree

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
    """
    A class to represent the membrane object.

    Attributes:
        ----------
        particle: class
            ParticleGeometry class
        mechanics: class
            MechanicalProperties class
        sampling_points_membrane: int
            number of points to sample the regions 2r and 2l

    Methods:
        -------
        compute_r2r_r2l_z2r_z2l_from_analytic_expression(self, f, particle, mechanics):
            Returns r and z coordinates in the regions 2r and 2l

    """

    def __init__(self, particle, sampling_points_membrane):
        """
        Constructs all the necessary attributes for the membrane object.

        Parameters:
            ----------
            particle: class
                ParticleGeometry class
            sampling_points_membrane: int
                number of points to sample the regions 2r and 2l

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
        """
        Computes the r and z coordinates to describe the regions 2r and 2l,
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
        sigma = mechanics.sigma_bar(f, wrapping)

        for i in range(1, len(self.S2)):
            s = self.S2[i]
            r = r2r_0 + s - sqrt(2 / sigma) * (1 - cos(alpha)) / (coth(s * sqrt(0.5 * sigma)) + cos(alpha * 0.5))
            z = z2r_0 + sqrt(8 / sigma) * sin(0.5 * alpha) * (
                1 - (csch(s * sqrt(0.5 * sigma))) / (coth(s * sqrt(0.5 * sigma)) + cos(0.5 * alpha))
            )
            r2r[i] = r
            z2r[i] = z

        r2r[0] = r2r_0
        z2r[0] = z2r_0
        r2l = np.array([r2r[0] - r2r[s] for s in range(len(self.S2))])
        z2l = z2r
        return r2r, z2r, r2l, z2l


class Wrapping:
    """
    A class to represent the wrapping of the particle.

    Attributes:
        ----------
        wrapíng_list: list
            list of wrapping degrees at which the system is evaluated

    """

    def __init__(self, wrapping_list):
        """
        Constructs all the necessary attributes for the membrane object.

        Parameters:
            ----------
            wrapíng_list: list
                list of wrapping degrees at which the system is evaluated

        Returns:
            -------
            None
        """
        self.wrapping_list = wrapping_list
