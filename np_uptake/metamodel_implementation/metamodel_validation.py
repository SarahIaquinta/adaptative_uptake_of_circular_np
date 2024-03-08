from tkinter import S

import numpy as np
import openturns as ot
import openturns.viewer as viewer
from scipy.interpolate import UnivariateSpline

ot.Log.Show(ot.Log.NONE)

import seaborn as sns
from matplotlib import pylab as plt

import np_uptake.metamodel_implementation.utils as miu
from np_uptake.figures.utils import CreateFigure, Fonts, SaveFigure, XTickLabels, XTicks
from np_uptake.metamodel_implementation.metamodel_creation import DataPreSetting, MetamodelPostTreatment


class MetamodelValidation:
    """A class that contains the methods to validate a metamodel

    Attributes:
    ----------
    None

    Methods:
    -------
    validate_metamodel_with_test(self, inputTest, outputTest, metamodel):
        Constructs a metamodel validator class (from the Openturns library)
    compute_Q2(self, metamodel_validator):
        Computes the predictivity factor of the metamodel.
    """

    def __init__(self):
        """Constructs all the necessary attributes for the MetamodelValidation object.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

    def validate_metamodel_with_test(self, inputTest, outputTest, metamodel):
        """Constructs a metamodel validator class (from the Openturns library).

        Parameters:
        ----------
        inputTest: class (array)
            Part of the dataset (input variables of the model) that will be used to validate
            the metamodel
        outputTest: class (array)
            Part of the dataset (output variables of the model) that will be used to validate
            the metamodel
        metamodel: class
            metamodel object (from the OpenTurns library)

        Returns:
        -------
        metamodel_validator: OT class
            Tool from the Openturns library used to validate a metamodel
        """

        metamodel_validator = ot.MetaModelValidation(inputTest, outputTest, metamodel)
        return metamodel_validator

    def compute_Q2(self, metamodel_validator):
        """Computes the predictivity factor of the metamodel.

        Parameters:
        ----------
        metamodel_validator: class
            Tool from the Openturns library used to validate a metamodel

        Returns:
        -------
        Q2: class (array)
            Predictivity factor
        """

        Q2 = metamodel_validator.computePredictivityFactor()
        return Q2

    def plot_prediction_vs_true_value_manual(
        self, type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    ):
        predicted_output = metamodel(inputTest)
        fig = createfigure.square_figure_7(pixels=pixels)
        ax = fig.gca()
        palette = sns.color_palette("Paired")
        orange = palette[-5]
        purple = palette[-3]
        color_plot = orange
        if type_of_metamodel == "Kriging":
            color_plot = purple
        ax.plot([0.05, 0.8], [0.05, 0.8], "-k", linewidth=2)
        ax.plot(outputTest, predicted_output, "o", color=color_plot)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_xticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_yticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((0, 0.83))
        ax.set_ylim((0, 0.83))
        ax.set_xlabel(r"true values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of $\Psi_3$", font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, type_of_metamodel + "_circular_" + str(pixels))


def metamodel_validation_routine_pce(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    degree,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    degree: int
        Dimension of the basis of polynoms


    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel + str(degree), training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data(shuffled_sample)

    gamma_bar_r_list_rescaled = miu.rescale_sample(inputTest[:, 0])
    gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(inputTest[:, 1])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(inputTest[:, 2])
    input_sample_Test_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 3)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_Test_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
        input_sample_Test_rescaled[k, 1] = gamma_bar_fs_bar_list_rescaled[k]
        input_sample_Test_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]

    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(
        input_sample_Test_rescaled, outputTest, metamodel
    )
    metamodelvalidation.plot_prediction_vs_true_value_manual(
        type_of_metamodel, input_sample_Test_rescaled, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )
    Q2 = metamodel_validator.computePredictivityFactor()
    residual, relative_error = metamodelposttreatment.get_errors_from_metamodel(results_from_algo)
    return Q2, residual, relative_error


def metamodel_validation_routine_kriging(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    type_of_metamodel,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Runs the routine to validate a metamodel:
        1 - imports the metamodel from a .pkl file
        2 - compares the true vs predicted value of the metamodel

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible value:
            "Kriging"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename = miu.create_pkl_name(type_of_metamodel, training_amount)
    shuffled_sample, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    inputTest, outputTest = datapresetting.extract_testing_data(shuffled_sample)

    metamodelvalidation.plot_prediction_vs_true_value_manual(
        type_of_metamodel, inputTest, outputTest, metamodel, createfigure, savefigure, xticks, pixels
    )


def plot_error_vs_degree_pce(
    degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
):
    """Plots the LOO error and the predictivity factor of the PCE with respect to its
        truncature degree

    Parameters:
    ----------
    degree_list: list
        List of the truncature degrees to be investigated
    Q2_list: list
        List of the predictivity factors obtained for each degree from degree_list
    relativeerror_list: list
        List of the LOO errors obtained for each degree from degree_list

    Returns:
    -------
    None
    """

    palette = sns.color_palette("Set2")
    vert_clair, orange = palette[0], palette[1]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.plot(degree_list, Q2_list, color=vert_clair, label="$Q_2$", linewidth=3)
    ax.plot(degree_list, relativeerror_list, color=orange, label=r"$\epsilon_{LOO}$", linewidth=3)
    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    plt.plot(degree_list[max_Q2_index], max_Q2, "o", color="k", markersize=15, mfc="none")
    min_relativeerror = min(relativeerror_list)
    min_relativeerror_index = relativeerror_list.index(min_relativeerror)
    plt.plot(degree_list[min_relativeerror_index], min_relativeerror, "o", color="k", markersize=15, mfc="none")
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_xticklabels(
        [1, 10, 20, 30, 40, 50],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(
        [0, 0.2, 0.4, 0.6, 0.8, 1],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax.set_xlim((0, 51))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xlabel("degree $p$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$Q_2$ [ - ], $\epsilon_{LOO}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "convergence_PCE_" + str(pixels))


def optimize_degree_pce(
    datapresetting,
    metamodelposttreatment,
    metamodelvalidation,
    degree_list,
    training_amount,
    createfigure,
    savefigure,
    xticks,
    pixels,
):
    """Determines the truncature degree of the PCE that maximizes the predictivity factor

    Parameters:
    ----------
    datapresetting: class
        A class that performs the presets on the dataset to compute the metamodel
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    metamodelvalidation: class
        Tool from the Openturns library used to validate a metamodel
    degree_list: list
        List of the truncature degrees to be investigated
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    Q2_list = []
    residual_list = []
    relativeerror_list = []
    for degree in degree_list:
        Q2, residual, relative_error = metamodel_validation_routine_pce(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "PCE",
            training_amount,
            degree,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )
        Q2_list.append(Q2[0])
        residual_list.append(residual[0])
        relativeerror_list.append(relative_error[0])

    plot_error_vs_degree_pce(
        degree_list, Q2_list, relativeerror_list, createfigure, savefigure, fonts, xticks, xticklabels, pixels
    )

    max_Q2 = max(Q2_list)
    max_Q2_index = Q2_list.index(max_Q2)
    optimal_degree = degree_list[max_Q2_index]
    print("Optimal degree for PCE: ", int(optimal_degree))


def plot_PDF_pce_kriging(metamodelposttreatment, degree, training_amount):
    """Plots the Probability Density Functions (PDFs) of the metamodel outputs and original data

    Parameters:
    ----------
    metamodelposttreatment: class from the metamodel_creation.py script
        A class that extracts and exports the features of the metamodel
    degree: int
        Dimension of the basis of polynoms
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel

    Returns:
    -------
    None
    """

    complete_pkl_filename_pce = miu.create_pkl_name("PCE" + str(degree), training_amount)
    shuffled_sample, results_from_algo_pce = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_pce)
    metamodel_pce = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_pce)
    complete_pkl_filename_kriging = miu.create_pkl_name("Kriging", training_amount)
    _, results_from_algo_kriging = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_kriging)
    metamodel_kriging = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo_kriging)
    gamma_bar_r_list_rescaled = miu.rescale_sample(shuffled_sample[:, 0])
    gamma_bar_fs_bar_list_rescaled = miu.rescale_sample(shuffled_sample[:, 1])
    gamma_bar_lambda_list_rescaled = miu.rescale_sample(shuffled_sample[:, 2])
    input_sample_rescaled = ot.Sample(len(gamma_bar_r_list_rescaled), 3)
    for k in range(len(gamma_bar_r_list_rescaled)):
        input_sample_rescaled[k, 0] = gamma_bar_r_list_rescaled[k]
        input_sample_rescaled[k, 1] = gamma_bar_fs_bar_list_rescaled[k]
        input_sample_rescaled[k, 2] = gamma_bar_lambda_list_rescaled[k]
    input_sample = ot.Sample(shuffled_sample[:, 0:3])
    output_model = shuffled_sample[:, -1]
    output_pce = metamodel_pce(input_sample_rescaled)
    output_kriging = metamodel_kriging(input_sample)

    def get_data_for_pdf(n, s):
        p, x = np.histogram(s, bins=n)  # bin it into n = N//10 bins
        x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
        f = UnivariateSpline(x, p, s=n)
        return x, f

    n_pdf = int(1e5)
    x_model, f_model = get_data_for_pdf(n_pdf, output_model)
    x_pce, f_pce = get_data_for_pdf(n_pdf, output_pce)
    x_kriging, f_kriging = get_data_for_pdf(n_pdf, output_kriging)

    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    ax.plot(x_model, f_model(x_model), "k", label="model", linewidth=2)
    ax.plot(x_kriging, f_kriging(x_kriging), color=purple, label="Kriging", linewidth=2)
    ax.plot(x_pce, f_pce(x_pce), color=orange, label="PCE", linewidth=2)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xticklabels(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0.005, 0.01, 0.015, 0.02])
    ax.set_yticklabels(
        ["0.5", "1.0", "1.5", "2.0"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlim((0, 0.83))
    ax.set_xlabel(r"$\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("PDF (x $10^{-2})$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    savefigure.save_as_png(fig, "PDF_metamodel_circular" + str(pixels))


if __name__ == "__main__":

    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    xticks = XTicks()
    xticklabels = XTickLabels()
    pixels = 360

    filename_qMC_Sobol = "dataset_for_metamodel_creation.txt"
    metamodelposttreatment = MetamodelPostTreatment()
    metamodelvalidation = MetamodelValidation()
    training_amount_list = [0.8]

    # degree_list = [10]#np.arange(1, 13)
    for training_amount in training_amount_list:
        datapresetting = DataPreSetting(filename_qMC_Sobol, training_amount)

    #     optimize_degree_pce(
    #         datapresetting,
    #         metamodelposttreatment,
    #         metamodelvalidation,
    #         degree_list,
    #         training_amount,
    #         createfigure,
    #         savefigure,
    #         xticks,
    #         pixels,
    #     )

        metamodel_validation_routine_kriging(
            datapresetting,
            metamodelposttreatment,
            metamodelvalidation,
            "Kriging",
            training_amount,
            createfigure,
            savefigure,
            xticks,
            pixels,
        )

    plot_PDF_pce_kriging(metamodelposttreatment, 10, 0.8)
