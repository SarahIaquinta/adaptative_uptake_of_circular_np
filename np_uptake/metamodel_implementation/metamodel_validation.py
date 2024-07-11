import openturns as ot
import openturns.viewer as viewer

ot.Log.Show(ot.Log.NONE)

from matplotlib import pylab as plt

import np_uptake.metamodel_implementation.utils as miu
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
    plot_prediction_vs_true_value(self, metamodel_validator, type_of_metamodel, training_amount):
        Plots the prediction (from metamodel) vs true value (of the model).
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

    def plot_prediction_vs_true_value(self, metamodel_validator, type_of_metamodel, training_amount):
        """Plots the prediction (from metamodel) vs true value (of the model).

        Parameters:
        ----------
        metamodel_validator: class
            Tool from the Openturns library used to validate a metamodel
        type_of_metamodel: string
            Name of the metamodel that has been computed. Possible values :
                "Kriging", "PCE"
        training_amount: float
            Proportion (between 0 and 1) of the initial data used for training (the remaining
            data are used for testing)

        Returns:
        -------
        None
        """

        graph = metamodel_validator.drawValidation()
        Q2 = metamodel_validator.computePredictivityFactor()
        graph.setTitle(
            type_of_metamodel
            + " metamodel validation (training amount = "
            + str(training_amount)
            + ") ; \n Q2 = "
            + str(Q2)
        )
        view = viewer.View(graph)
        plt.xlabel(r"true value of $\tilde{f}$ (from model)")
        plt.ylabel(r"predicted value of $\tilde{f}$ (from metamodel)")
        plt.show()


def metamodel_validation_routine(
    datapresetting, metamodelposttreatment, metamodelvalidation, type_of_metamodel, training_amount
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
        Name of the metamodel that has been computed. Possible values :
            "Kriging", "PCE"
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
    metamodel_validator = metamodelvalidation.validate_metamodel_with_test(inputTest, outputTest, metamodel)
    metamodelvalidation.plot_prediction_vs_true_value(metamodel_validator, type_of_metamodel, training_amount)


if __name__ == "__main__":
    filename_qMC_Sobol = "dataset_for_metamodel_creation.txt"
    metamodelposttreatment = MetamodelPostTreatment()
    metamodelvalidation = MetamodelValidation()
    training_amount_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for training_amount in training_amount_list:
        datapresetting = DataPreSetting(filename_qMC_Sobol, training_amount)
        metamodel_validation_routine(
            datapresetting, metamodelposttreatment, metamodelvalidation, "Kriging", training_amount
        )
        metamodel_validation_routine(
            datapresetting, metamodelposttreatment, metamodelvalidation, "PCE", training_amount
        )
