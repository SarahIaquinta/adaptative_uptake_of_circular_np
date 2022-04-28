from pathlib import Path

import numpy as np
import openturns as ot

ot.Log.Show(ot.Log.NONE)
import metamodel_implementation.utils


class DataPreSetting:
    """
    A class that performs the presets on the dataset to compute the metamodel

    Attributes:
        ----------
        filename: string
            name of the .txt file from which the data will be extracted
        training_amount : float
            proportion (between 0 and 1) of the initial data used for training (the remaining data
            are used for testing)

    Methods:
        -------
        shuffle_dataset_from_datafile(self):
            imports and shuffles the initial dataset
        extract_training_data_from_shuffled_dataset(self, shuffled_sample):
            extracts a proportion (training_amount) of the shuffled_data to generate the training
            dataset

    """

    def __init__(self, filename, training_amount):
        """
        Constructs all the necessary attributes for the DataPreSetting object.

        Parameters:
            ----------
            filename: string
                name of the .txt file from which the data will be extracted
            training_amount : float
                proportion (between 0 and 1) of the initial data used for training (the remaining
                data are used for testing)

        Returns:
            -------
            None
        """

        self.filename = Path.cwd() / "metamodel_implementation" / filename
        self.training_amount = training_amount

    def shuffle_dataset_from_datafile(self):
        """
        imports and shuffles the initial dataset using Openturns library

        Parameters:
            ----------
            None

        Returns:
            -------
            sample: ot class
                shuffled input dataset

        """
        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        np.random.shuffle(sample)
        return sample

    def extract_training_data_from_shuffled_dataset(self, shuffled_sample):
        """
        extracts a proportion (training_amount) of the shuffled_data to train the metamodel

        Parameters:
            ----------
            shuffled_sample : ot class


        Returns:
            -------
            shuffled_training_input_sample: ot class
                part of the input data used to train the metamodel
            shuffled_training_output_sample: ot class
                output data corresponding to the shuffled_training_input_sample

        """
        datasize, _ = np.shape(shuffled_sample)
        datasize = 1000
        shuffled_training_input_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), 0:3])
        shuffled_training_output_sample = ot.Sample(shuffled_sample[: int(self.training_amount * datasize), -1:])
        return shuffled_training_input_sample, shuffled_training_output_sample

    def extract_testing_data(self, shuffled_sample):
        """
        extracts a proportion (1 - training_amount) of the shuffled_data to test the metamodel

        Parameters:
            ----------
            shuffled_sample : ot class


        Returns:
            -------
            shuffled_testing_input_sample: ot class
                part of the input data used to test the metamodel
            shuffled_testing_output_sample: ot class
                output data corresponding to the shuffled_testing_input_sample

        """
        datasize, _ = np.shape(shuffled_sample)
        datasize = 1000
        shuffled_testing_input_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, 0:3]
        )
        shuffled_testing_output_sample = ot.Sample(
            shuffled_sample[int(self.training_amount * datasize + 1) : datasize, -1:]
        )
        return shuffled_testing_input_sample, shuffled_testing_output_sample


class MetamodelCreation:
    def __init__(self, input_sample_training, output_sample_training):
        """
        Constructs all the necessary attributes for the MetamodelCreation object.

        Parameters:
            ----------
            input_sample_training: ot class (array)
                sample of input parameters for training, generated in the DataPreSetting class
            output_sample_training: ot class (array)
                sample of output parameters for training, generated in the DataPreSetting class

        Returns:
            -------
            None
        """

        self.input_sample_training = input_sample_training
        self.output_sample_training = output_sample_training
        _, self.dimension = np.shape(self.input_sample_training)

    def create_kriging_algorithm(self):
        """
        computes the Kriging Algorithm of the Openturns library

        Parameters:
            ----------
            None


        Returns:
            -------
            kriging_algorithm: ot class
                Kriging algorithm from the Openturns library

        """
        basis = ot.Basis(0)
        cov = ot.SphericalModel(self.dimension)
        kriging_algorithm = ot.KrigingAlgorithm(self.input_sample_training, self.output_sample_training, cov, basis)
        return kriging_algorithm


class MetamodelPostTreatment:
    def __init__(self):
        """
        Constructs all the necessary attributes for the MetamodelPostTreatment object.

        Parameters:
            ----------
            None

        Returns:
            -------
            None
        """

    def run_algorithm(self, algorithm):
        """
        runs the algorithm

        Parameters:
            ----------
            algorithm: ot class
                algorithm (kriging), output of the methods which creates the respective algorithm


        Returns:
            -------
            Nothing

        """
        algorithm.run()

    def extract_results_from_algorithm(self, algorithm):
        """
        gets the results from the algorithm

        Parameters:
            ----------
            algorithm: ot class
                algorithm (kriging), output of the methods which creates the respective algorithm


        Returns:
            -------
            results: ot class
                result from the algorithm, computing following a method from the Openturns library

        """
        results = algorithm.getResult()
        return results

    def get_metamodel_from_results_algo(self, results_from_algo):
        """
        gets the metamodel from the algorithm's results

        Parameters:
            ----------
            results_from_algo: ot class
                result from the algorithm, computing following a method from the Openturns library
                output of the method self.extract_results_from_algorithm(algorithm)


        Returns:
            -------
            metamodel: ot class
                metamodel computed by the algorithm. contains the information necessary to use the
                metamodel

        """
        metamodel = results_from_algo.getMetaModel()
        return metamodel

    def get_errors_from_metamodel(self, results_from_algo):
        """
        gets the errors from the metamodel (obtained according to the results_from_algo)

        Parameters:
            ----------
            results_from_algo: ot class
                result from the algorithm, computing following a method from the Openturns library


        Returns:
            -------
            residual: ot class (array)
                residual of the metamodel
            relative_error: ot class (array)
                relative_error of the metamodel

        """

        residual = results_from_algo.getResiduals()
        relative_error = results_from_algo.getRelativeErrors()
        return residual, relative_error


def metamodel_creation_routine_kriging(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample):
    """
    Runs the routine to create a metamodel:
        1 - imports the data that will be used to create the metamodel 2 - creates the metamodel 3
        - exports the metamodel in a .pkl file

    Parameters:
        ----------
        datapresetting: class
            A class that performs the presets on the dataset to compute the metamodel
        metamodelcreation: class
            A class that contains the routines to create a metamodel
        metamodelposttreatment: class
            A class that contains the routines to extract the algorithms from the metamodel
        shuffled_sample: ot class
            sample that was used to create the metamodel

    Returns:
        -------
        Nothing

    """
    kri = metamodelcreation.create_kriging_algorithm()
    metamodelposttreatment.run_algorithm(kri)
    results_from_kri = metamodelposttreatment.extract_results_from_algorithm(kri)
    complete_pkl_filename_Kriging = metamodel_implementation.utils.create_pkl_name(
        "Kriging", datapresetting.training_amount
    )
    metamodel_implementation.utils.export_metamodel_and_data_to_pkl(
        shuffled_sample, results_from_kri, complete_pkl_filename_Kriging
    )


if __name__ == "__main__":
    filename_qMC_Sobol = "dataset_for_metamodel_creation.txt"
    training_amount_list = [0.7]  # , 0.75, 0.8, 0.85, 0.9, 0.95]
    metamodelposttreatment = MetamodelPostTreatment()
    for training_amount in training_amount_list:
        datapresetting = DataPreSetting(filename_qMC_Sobol, training_amount)
        shuffled_sample = datapresetting.shuffle_dataset_from_datafile()
        input_sample_training, output_sample_training = datapresetting.extract_training_data_from_shuffled_dataset(
            shuffled_sample
        )
        metamodelcreation = MetamodelCreation(input_sample_training, output_sample_training)
        metamodel_creation_routine_kriging(datapresetting, metamodelcreation, metamodelposttreatment, shuffled_sample)
