import openturns as ot
import openturns.viewer as viewer

ot.Log.Show(ot.Log.NONE)


import numpy as np

import np_uptake.metamodel_implementation.utils as miu
from np_uptake.figures.utils import CreateFigure, Fonts, SaveFigure
from np_uptake.metamodel_implementation.metamodel_validation import MetamodelPostTreatment


class Distribution:
    """A class that gathers the distributions of the input variables

    Attributes:
    ----------
    None

    Methods:
    -------
    uniform(self):
        Creates a uniform distribution of the 3 input parameters
    """

    def __init__(self):
        """Constructs all the necessary attributes for the Distribution object.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        self.gamma_bar_r_min = 1
        self.gamma_bar_r_max = 6
        self.gamma_bar_fs_min = -0.45
        self.gamma_bar_fs_max = 0.45
        self.gamma_bar_lambda_min = 10
        self.gamma_bar_lambda_max = 100

    def uniform(self):
        """Creates a uniform distribution of the 3 input parameters

        Parameters:
        ----------
        None

        Returns:
        -------
        distribution: ot class
            Uniform distribution of the 3 input parameters, computed wth Openturns.
            gamma_bar_lambda could have been computed as constant values
            but we chose to generate it as uniform distribution with close bounds to match
            the architecture of the Openturns library
        """

        distribution = ot.ComposedDistribution(
            [
                ot.Uniform(self.gamma_bar_r_min, self.gamma_bar_r_max),
                ot.Uniform(self.gamma_bar_fs_min, self.gamma_bar_fs_max),
                ot.Uniform(self.gamma_bar_lambda_min, self.gamma_bar_lambda_max),
            ]
        )
        return distribution


# Saltelli#
def compute_sensitivity_algo_Saltelli(distribution, metamodel, sensitivity_experiment_size):
    """Computes the sensitivity algorithms computed after the Saltelli method

    Parameters:
    ----------
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested
        in the sensitivity analysis
    metamodel: ot class
        Metamodel (Kriging or PCE) computed in the metamodel_creation.py script and stored in a .pkl
        file
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
    -------
    sensitivityAnalysis: ot class
        Sensitivity algorithm computed with the Openturns class
    """

    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(myExperiment, metamodel, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_Saltelli(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="Saltelli",
):
    """Computes the sensitivity algorithms computed after the Saltelli method and exports it to a
    .pkl file

    Parameters:
    ----------
    type_of_metamodel: str
        Type of metamodel that has been computed. Possible values: "Kriging", "PCE"
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data
        are used for testing)
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested
        in the sensitivity analysis
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity
    type_of_Sobol_sensitivity_implementation: str
        Type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
        sensitivity algorithm is stored.

    Returns:
    -------
    None
    """

    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel, training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_Saltelli = compute_sensitivity_algo_Saltelli(distribution, metamodel, sensitivity_experiment_size)
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(
        type_of_metamodel, training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_Saltelli, complete_pkl_filename_sensitivy_algo)


# Jansen#
def compute_sensitivity_algo_Jansen(distribution, metamodel, sensitivity_experiment_size):
    """Computes the sensitivity algorithms computed after the Jansen method

    Parameters:
    ----------
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested
        in the sensitivity analysis
    metamodel: ot class
        Metamodel (Kriging or PCE) computed in the metamodel_creation.py script and stored in a .pkl
        file
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
    -------
    sensitivityAnalysis: ot class
        Sensitivity algorithm computed with the Openturns class
    """

    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    sensitivityAnalysis = ot.JansenSensitivityAlgorithm(myExperiment, metamodel, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_Jansen(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="Jansen",
):
    """Computes the sensitivity algorithms computed after the Jansen method and exports it to a
    .pkl file

    Parameters:
    ----------
    type_of_metamodel: str
        Type of metamodel that has been computed. Possible values: "Kriging", "PCE"
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data
        are used for testing)
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested
        in the sensitivity analysis
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity
    type_of_Sobol_sensitivity_implementation: str
        Type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
        sensitivity algorithm is stored.

    Returns:
    -------
    None
    """

    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel, training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_Jansen = compute_sensitivity_algo_Jansen(distribution, metamodel, sensitivity_experiment_size)
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(
        type_of_metamodel, training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_Jansen, complete_pkl_filename_sensitivy_algo)


# MauntzKucherenko#
def compute_sensitivity_algo_MauntzKucherenko(distribution, metamodel, sensitivity_experiment_size):
    """Computes the sensitivity algorithms computed after the MauntzKucherenko method

    Parameters:
    ----------
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested
        in the sensitivity analysis
    metamodel: ot class
        Metamodel (Kriging or PCE) computed in the metamodel_creation.py script and stored in a .pkl
        file
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
    -------
    sensitivityAnalysis: ot class
        Sensitivity algorithm computed with the Openturns class
    """

    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    sensitivityAnalysis = ot.MauntzKucherenkoSensitivityAlgorithm(myExperiment, metamodel, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_MauntzKucherenko(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="MauntzKucherenko",
):
    """Computes the sensitivity algorithms computed after the Mauntz-Kucherenko method and exports
    it to a .pkl file

    Parameters:
    ----------
    type_of_metamodel: str
        Type of metamodel that has been computed. Possible values: "Kriging", "PCE"
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data are
        used for testing)
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested in
        the sensitivity analysis
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity
    type_of_Sobol_sensitivity_implementation: str
        Type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
        sensitivity algorithm is stored.

    Returns:
    -------
    None
    """

    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel, training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_MauntzKucherenko = compute_sensitivity_algo_MauntzKucherenko(
        distribution, metamodel, sensitivity_experiment_size
    )
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(
        type_of_metamodel, training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_MauntzKucherenko, complete_pkl_filename_sensitivy_algo)


# Martinez#
def compute_sensitivity_algo_Martinez(distribution, metamodel, sensitivity_experiment_size):
    """Computes the sensitivity algorithms computed after the Martinez method

    Parameters:
    ----------
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested
        in the sensitivity analysis
    metamodel: ot class
        Metamodel (Kriging or PCE) computed in the metamodel_creation.py script and stored in a .pkl
        file
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
    -------
    sensitivityAnalysis: ot class
        Sensitivity algorithm computed with the Openturns class
    """

    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(myExperiment, metamodel, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_Martinez(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="Martinez",
):
    """Computes the sensitivity algorithms computed after the Martinez method and exports it to a
    .pkl file

    Parameters:
    ----------
    type_of_metamodel: str
        Type of metamodel that has been computed. Possible values: "Kriging", "PCE"
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data
        are used for testing)
    distribution: class
        Class that defines the distribution of the input parameters whose influence is tested
        in the sensitivity analysis
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity
    type_of_Sobol_sensitivity_implementation: str
        Type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
        sensitivity algorithm is stored.

    Returns:
    -------
    None
    """

    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel, training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_Martinez = compute_sensitivity_algo_Martinez(distribution, metamodel, sensitivity_experiment_size)
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(
        type_of_metamodel, training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_Martinez, complete_pkl_filename_sensitivy_algo)


# Sobol indices from PCE#
def compute_sensitivity_indices_PCE(training_amount, degree):
    """Computes the sensitivity algorithms directly from the coefficients of the PCE metamodel

    Parameters:
    ----------
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data are
        used for testing)
    degree: float
        truncature degree of the PCE metamodel

    Returns:
    -------
    first_order_indices: list
        First order Sobol indices. Position in the list follows the position in the datafile:
        gamma_bar_r, gamma_bar_fs, gamma_bar_lambda
    total_order_indices: list
        Total order Sobol indices. Position in the list follows the position in the datafile:
        gamma_bar_r, gamma_bar_fs, gamma_bar_lambda
    """

    complete_pkl_filename_pce = miu.create_pkl_name("PCE" + str(degree), training_amount)
    [_, results_from_algo] = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    chaosSI = ot.FunctionalChaosSobolIndices(results_from_algo)
    first_order_indices = [chaosSI.getSobolIndex(k) for k in range(3)]
    total_order_indices = [chaosSI.getSobolTotalIndex(k) for k in range(3)]
    return first_order_indices, total_order_indices


# PLOTS#
def plot_results_sensitivity_analysis(
    type_of_metamodel,
    training_amount,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation,
    createfigure,
    pixels,
):
    """Plots the first and total Sobol indices

    Parameters:
    ----------
    type_of_metamodel: str
        Type of metamodel that has been computed. Possible values: "Kriging", "PCE"
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data
        are used for testing)
    sensitivity_experiment_size: float (int)
        Amount of estimations of the metamodel to evaluate the sensitivity
    type_of_Sobol_sensitivity_implementation: str
        Type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
        sensitivity algorithm is stored.
    createfigure: class
        Class from the figures.utils.py script that provides a predefined figure layout
    colors: class
        Class from the figures.utils.py script that provides a predefined set of colors
    pixels: str
        Number of points per pixel in the figures Recommended: 360

    Returns:
    -------
    None
    """

    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(
        type_of_metamodel, training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    sensitivity_algo = miu.extract_sensitivity_algo_from_pkl(complete_pkl_filename_sensitivy_algo)
    first_order_indices = sensitivity_algo.getFirstOrderIndices()
    first_order_indices_confidence_interval = sensitivity_algo.getFirstOrderIndicesInterval()
    first_order_indices_confidence_lowerbounds = [
        first_order_indices_confidence_interval.getLowerBound()[k] for k in [0, 1, 2]
    ]
    first_order_indices_confidence_upperbounds = [
        first_order_indices_confidence_interval.getUpperBound()[k] for k in [0, 1, 2]
    ]
    total_order_indices = sensitivity_algo.getTotalOrderIndices()
    total_order_indices_confidence_interval = sensitivity_algo.getTotalOrderIndicesInterval()
    total_order_indices_confidence_lowerbounds = [
        total_order_indices_confidence_interval.getLowerBound()[k] for k in [0, 1, 2]
    ]
    total_order_indices_confidence_upperbounds = [
        total_order_indices_confidence_interval.getUpperBound()[k] for k in [0, 1, 2]
    ]
    first_order_indices_confidence_errorbar = np.zeros((2, 3))
    total_order_indices_confidence_errorbar = np.zeros((2, 3))
    for k in range(2):
        first_order_indices_confidence_errorbar[0, k] = (
            first_order_indices[k] - first_order_indices_confidence_lowerbounds[k]
        )
        first_order_indices_confidence_errorbar[1, :] = (
            first_order_indices_confidence_upperbounds[k] - first_order_indices[k]
        )
        total_order_indices_confidence_errorbar[0, :] = (
            total_order_indices[k] - total_order_indices_confidence_lowerbounds[k]
        )
        total_order_indices_confidence_errorbar[1, :] = (
            total_order_indices_confidence_upperbounds[k] - total_order_indices[k]
        )

    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.errorbar(
        [0, 1, 2],
        first_order_indices,
        yerr=first_order_indices_confidence_errorbar,
        label="First order indice",
        color="k",
        marker="v",
        markersize=12,
        linestyle="None",
    )
    ax.errorbar(
        [0, 1, 2],
        total_order_indices,
        yerr=total_order_indices_confidence_errorbar,
        label="Total indice",
        color="m",
        marker="D",
        markersize=12,
        linestyle="None",
    )
    ax.set_xlim((-0.2, 2.2))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        [r"$\overline{\gamma}_A$", r"$\overline{\gamma}_D$", r"$\overline{\gamma}_R$"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(
        ["0", "0.25", "0.5", "0.75", "1"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlabel("Variable", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Sobol indice [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    filename = (
        "sobolindices_algo="
        + type_of_Sobol_sensitivity_implementation
        + "_metamodel="
        + type_of_metamodel
        + "_trainingamount="
        + str(training_amount)
        + "_size="
        + str(sensitivity_experiment_size)
        + "_"
    )
    savefigure.save_as_png(fig, filename + str(pixels) + "p")


def plot_sensitivity_indices_PCE(
    training_amount,
    degree,
    createfigure,
    pixels,
):
    """Plots the first and total Sobol indice obtained from PCE metamodel

    Parameters:
    ----------
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data
        are used for testing)
    degree: float
        truncature degree of the PCE metamodel
    createfigure: class
        Class from the figures.utils.py script that provides a predefined figure layout
    colors: class
        Class from the figures.utils.py script that provides a predefined set of colors
    pixels: str
        Number of points per pixel in the figures Recommended: 360

    Returns:
    -------
    None
    """

    first_order_indices, total_order_indices = compute_sensitivity_indices_PCE(training_amount, degree)
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.plot(
        [0, 1, 2],
        first_order_indices,
        label="First order indice",
        color="k",
        marker="v",
        markersize=12,
        linestyle="None",
    )
    ax.plot(
        [0, 1, 2],
        total_order_indices,
        label="Total indice",
        color="m",
        marker="D",
        markersize=12,
        linestyle="None",
    )
    ax.set_xlim((-0.2, 2.2))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        [r"$\overline{\gamma}_A$", r"$\overline{\gamma}_D$", r"$\overline{\gamma}_R$"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(
        ["0", "0.25", "0.5", "0.75", "1"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlabel("Variable", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Sobol indice [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    filename = "sobolindices_analyticalfromPCE_trainingamount=" + str(training_amount) + "_"
    savefigure.save_as_png(fig, filename + str(pixels) + "p")


if __name__ == "__main__":
    type_of_metamodel = "Kriging"
    training_amount = 0.7
    sensitivity_experiment_size_list = [10000]
    type_of_Sobol_sensitivity_implementation_list = ["Saltelli", "Jansen", "MauntzKucherenko", "Martinez"]
    metamodelposttreatment = MetamodelPostTreatment()
    distribution = Distribution()
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    training_amount_list = [0.8]
    degree = 10
    for training_amount in training_amount_list:
        plot_sensitivity_indices_PCE(training_amount, degree, createfigure, pixels=360)
    for sensitivity_experiment_size in sensitivity_experiment_size_list:
        compute_and_export_sensitivity_algo_Saltelli(
            type_of_metamodel, training_amount, distribution, sensitivity_experiment_size
        )
        compute_and_export_sensitivity_algo_Jansen(
            type_of_metamodel, training_amount, distribution, sensitivity_experiment_size
        )
        compute_and_export_sensitivity_algo_MauntzKucherenko(
            type_of_metamodel, training_amount, distribution, sensitivity_experiment_size
        )
        compute_and_export_sensitivity_algo_Martinez(
            type_of_metamodel, training_amount, distribution, sensitivity_experiment_size
        )
        for type_of_Sobol_sensitivity_implementation in type_of_Sobol_sensitivity_implementation_list:
            plot_results_sensitivity_analysis(
                type_of_metamodel,
                training_amount,
                sensitivity_experiment_size,
                type_of_Sobol_sensitivity_implementation,
                createfigure,
                pixels=360,
            )
