import pickle
from pathlib import Path


def create_pkl_name(type_of_metamodel, training_amount, folder=""):
    """Creates the name of the .pkl file in which the metamodel will be stored

    Parameters:
    ----------
    type_of_metamodel: string
        Name of the metamodel that has been computed.
        Possible values :
            "Kriging", "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    folder: string
        The name of the folder in which the .pkl will be created and saved
        default value: ""

    Returns:
    -------
    complete_filename: str
        Name of the .pkl file
    """

    path = Path.cwd() / folder
    pkl_name = "metamodel_" + type_of_metamodel + "_trainingamount_" + str(training_amount) + ".pkl"
    complete_filename = path / pkl_name
    return complete_filename


def create_pkl_name_sensitivityalgo(
    type_of_metamodel,
    training_amount,
    experiment_size,
    sobol_implementation,
    folder="sensitivity_analysis",
):
    """Creates the name of the .pkl file in which the sensitivity algorithm will be stored

    Parameters:
    ----------
    type_of_metamodel: string
        Name of the metamodel that has been computed. Possible values :
            "Kriging", "PCE"
    training_amount: float (between 0 and 1)
        Amount of the data that is used to train the metamodel
    experiment_size: float
        Number of simulations used to compute the sensitivity algorithm
    sobol_implementation: string
        Name of the Sobol algorithm implemented Possible values :
            "Jansen", "Martinez", "MauntzKucherenko", "Saltelli"
    folder: string
        The name of the folder in which the .pkl will be created and saved default value:
        "sensitivity_analysis"

    Returns:
    -------
    complete_filename: str
        Name of the .pkl file

    """
    path = Path.cwd() / folder
    pkl_name = (
        "sensitivityalgo="
        + sobol_implementation
        + "_size="
        + str(experiment_size)
        + "_metamodel="
        + type_of_metamodel
        + "_trainingamount="
        + str(training_amount)
        + ".pkl"
    )
    complete_filename = path / pkl_name
    return complete_filename


def extract_metamodel_and_data_from_pkl(complete_filename):
    """Extracts the objects that have been stored with the metamodel .pkl

    Parameters:
    ----------
    complete_filename: str
        Name of the .pkl file

    Returns:
    -------
    sample: ot.class
        Input dataset used to create the metamodel
    results_from_algo: ot.class
        Class which possesses all the information relative to the metamodel that has been generated
    """

    with open(complete_filename, "rb") as f:
        [sample, results_from_algo] = pickle.load(f)
    return sample, results_from_algo


def export_metamodel_and_data_to_pkl(sample, results_from_algo, complete_filename):
    """Exports the objects to the metamodel .pkl

    Parameters:
    ----------
    sample: ot.class
        Input dataset used to create the metamodel
    results_from_algo: ot.class
        Class which possesses all the information relative to the metamodel that has been generated
    complete_filename: str
        Name of the .pkl file

    Returns:
    -------
    None
    """

    with open(complete_filename, "wb") as f:
        pickle.dump(
            [sample, results_from_algo],
            f,
        )


def export_sensitivity_algo_to_pkl(sensitivity_algo, complete_filename):
    """Exports the objects to the sensitivity .pkl

    Parameters:
    ----------
    sensitivity_algo: ot.class
        Sensitivity algorithm
    complete_filename: str
        Name of the .pkl file

    Returns:
    -------
    None
    """

    with open(complete_filename, "wb") as f:
        pickle.dump(
            [sensitivity_algo],
            f,
        )


def extract_sensitivity_algo_from_pkl(complete_filename):
    """Extracts the objects from the sensitivity .pkl

    Parameters:
    ----------
    complete_filename: str
        Name of the .pkl file

    Returns:
    -------
    sensitivity_algo: ot.class
        Sensitivity algorithm
    """

    with open(complete_filename, "rb") as f:
        [sensitivity_algo] = pickle.load(f)
    return sensitivity_algo


def rescale_sample(vector):
    """Rescales a vector between -1 and 1

    Parameters:
    ----------
    vector: vector object
        vector to be rescaled

    Returns:
    -------
    vector_normalized: list
        normalized vector
    """

    vector_start0 = [k[0] - vector.getMin()[0] for k in vector]
    vector_end2 = [k * 2 / max(vector_start0) for k in vector_start0]
    vector_normalized = [k - 1 for k in vector_end2]
    return vector_normalized
