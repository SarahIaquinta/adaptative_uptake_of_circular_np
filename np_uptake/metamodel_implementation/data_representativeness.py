import pickle
from pathlib import Path

import numpy as np
import openturns as ot
from sklearn.neighbors import KernelDensity
import seaborn as sns
ot.Log.Show(ot.Log.NONE)

from np_uptake.figures.utils import CreateFigure, Fonts, SaveFigure


class SampleRepresentativeness:
    """A class that investigates the representativeness of the dataset

    Attributes:
    ----------
    filename: string
        Name of the .txt file from which the data will be extracted
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data
        are used for testing)
    nb_of_shuffled_samples: float
        Number of shuffled samples of the dataset, used to evaluate the standard deviation of
        the cumulative mean of a sample
    pixels: string
        Number of points per pixel in the figures. Recommended: 360

    Methods:
    -------
    generate_shuffled_samples(self):
        Shuffles the dataset output (phase 3) self.nb_of_shuffled_samples times
    compute_cumulative_mean_std(self, vector):
        Computes the cumulative mean and standard deviation (std) of a vector
    compute_means_stds_of_shuffled_samples_and_export_to_pkl(self):
        Computes the cumulative mean and standard deviation (std) for the
        self.nb_of_shuffled_samples shuffled samples that have been generated Exports them into
        a .pkl file
    plot_cumulative_mean_vs_sample_size(self, createfigure, savefigure, fonts):
        Plots the cumulative mean of a sample with the std (computed from the
        self.nb_of_shuffled_samples shuffled samples)
    plot_cumulative_std_vs_sample_size(self, createfigure, savefigure, fonts):
        Plots the cumulative std of a sample with the std (computed from the
        self.nb_of_shuffled_samples shuffled samples)
    plot_gradient_cumulative_mean_vs_sample_size(self, createfigure, savefigure, fonts):
        Plots the absolute gradient of the cumulative mean of a sample
    plot_gradient_cumulative_std_vs_sample_size(self, createfigure, savefigure, fonts):
        Plots the absolute gradient of the cumulative std of a sample
    """

    def __init__(self, filename, training_amount, nb_of_shuffled_samples, pixels):
        """Constructs all the necessary attributes for the SampleRepresentativeness object.

        Parameters:
        ----------
        filename: string
            Name of the .txt file from which the data will be extracted
        training_amount: float
            Proportion (between 0 and 1) of the initial data used for training (the remaining
            data are used for testing)
        nb_of_shuffled_samples: float
            Number of shuffled samples of the dataset, used to evaluate the standard deviation
            of the cumulative mean of a sample
        pixels: string
            Number of points per pixel in the figures. Recommended: 360

        Returns:
        -------
        None
        """

        self.filename = Path.cwd() / "np_uptake/metamodel_implementation" / filename
        self.training_amount = training_amount
        self.pixels = pixels
        self.nb_of_shuffled_samples = nb_of_shuffled_samples

    def generate_shuffled_samples(self):
        """Shuffles the dataset output (phase 3) self.nb_of_shuffled_samples times

        Parameters:
        ----------
        None

        Returns:
        -------
        all_shuffled_phase3: array of shape ((len(sample) , self.nb_of_shuffled_samples))
            self.nb_of_shuffled_samples times shuffled phase 3
        """

        sample = ot.Sample.ImportFromTextFile(self.filename.as_posix(), "\t", 0)
        phase3 = sample[:, -1]
        all_shuffled_phase3 = np.zeros((len(phase3), self.nb_of_shuffled_samples))
        for i in range(self.nb_of_shuffled_samples):
            np.random.shuffle(phase3)
            for k in range(len(phase3)):
                all_shuffled_phase3[k, i] = phase3[k, 0]
        return all_shuffled_phase3

    def plot_PDF_phase3(self, createfigure, savefigure, fonts):
        proportion_phase_3_list = self.generate_shuffled_samples()[:, 0]
        fig = createfigure.square_figure_7(pixels=360)
        ax = fig.gca()
        X_plot_phase3 = np.linspace(0, 1, len(proportion_phase_3_list))[:, None].reshape(1, -1)
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.03).fit(proportion_phase_3_list.reshape(1, -1))
        log_dens_model = kde_model.score_samples(X_plot_phase3)
        ax.hist(proportion_phase_3_list, bins=20, density=True, color="lightgray", alpha=0.3, ec="black")
        ax.plot(
            X_plot_phase3[:, 0],
            np.exp(log_dens_model),
            color='k',
            lw=2,
            linestyle="-", label="model",
        )
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels(
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        ax.set_yticklabels(
            [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlim((-0.02, 1.02))
        ax.set_xlabel(r"$\Psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"$p_{\Psi_3}(\psi_3)$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
        ax.grid(linestyle="--")
        savefigure.save_as_png(fig, "PDF_psi3")

    def compute_cumulative_mean_std(self, vector):
        """Computes the cumulative mean and standard deviation (std) of a vector

        Parameters:
        ----------
        vector: array
            Vector of which the cumulative mean and std will be computed

        Returns:
        -------
        cumulative_mean: array, same shape as vector
            Cumulative mean of the vector
        cumulative_std: array, same shape as vector
            Cumulative std of the vector
        """

        cumulative_mean = np.zeros(len(vector))
        cumulative_std = np.zeros_like(cumulative_mean)
        for i in range(len(vector)):
            cumulative_mean[i] = np.mean(vector[0:i])
            cumulative_std[i] = np.std(vector[0:i])
        return cumulative_mean, cumulative_std

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl(self):
        """Computes the cumulative mean and standard deviation (std) for the
            self.nb_of_shuffled_samples shuffled samples that have been generated.
            Exports them into a .pkl file named "data_representativeness.pkl"

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        Exports:
        -------
        cumulative_mean: array
            Cumulative mean of one shuffled sample
        std_of_cumulative_means: array
            Std of the cumulative_mean of all the shuffled samples
        cumulative_std: array
            Cumulative std of one shuffled sample
        std_of_cumulative_stds: array
            Std of the cumulative_std of all the shuffled samples
        all_shuffled_phase3: array
            Shuffled samples used in this method.
            Output of the self.generate_shuffled_samples() method
        """

        all_shuffled_phase3 = self.generate_shuffled_samples()
        cumulative_means_for_all_samples = np.zeros_like(all_shuffled_phase3)
        cumulative_stds_for_all_samples = np.zeros_like(all_shuffled_phase3)

        std_of_cumulative_means = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        cumulative_std = np.zeros_like(cumulative_means_for_all_samples[:, 0])
        std_of_cumulative_stds = np.zeros_like(cumulative_std)
        for i in range(self.nb_of_shuffled_samples):
            cumulative_mean, cumulative_std = self.compute_cumulative_mean_std(all_shuffled_phase3[:, i])
            cumulative_means_for_all_samples[:, i] = cumulative_mean
            cumulative_stds_for_all_samples[:, i] = cumulative_std

        for j in range(len(std_of_cumulative_means)):
            std_of_cumulative_means[j] = np.std(cumulative_means_for_all_samples[j, :])
            std_of_cumulative_stds[j] = np.std(cumulative_stds_for_all_samples[j, :])

        with open("data_representativeness.pkl", "wb") as f:
            pickle.dump(
                [cumulative_mean, std_of_cumulative_means, cumulative_std, std_of_cumulative_stds, all_shuffled_phase3],
                f,
            )

    def plot_cumulative_mean_vs_sample_size(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """Plots the cumulative mean of a sample with the std (computed from the
            self.nb_of_shuffled_samples shuffled samples)

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        with open("data_representativeness.pkl", "rb") as f:
            [cumulative_mean, std_of_cumulative_means, _, _, all_shuffled_phase3] = pickle.load(f)

        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_phase3[:, 0])[0],
            std_of_cumulative_means,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 250, 500, 750, 1000])
        ax.set_xticklabels(
            [1, 250, 500, 750, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(-0.02, 0.42)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticklabels(
            [0, 0.1, 0.2, 0.3, 0.4],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Sample size [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Mean of $\psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid()
        savefigure.save_as_png(fig, "cumulative_mean_vs_sample_size")

    def plot_cumulative_std_vs_sample_size(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """Plots the cumulative std of a sample with the std (computed from the
           self.nb_of_shuffled_samples shuffled samples)

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        with open("data_representativeness.pkl", "rb") as f:
            [_, _, cumulative_std, std_of_cumulative_stds, all_shuffled_phase3] = pickle.load(f)
        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        ax.errorbar(
            sample_size,
            self.compute_cumulative_mean_std(all_shuffled_phase3[:, 0])[1],
            std_of_cumulative_stds,
            color="black",
            ecolor="gray",
            elinewidth=0.5,
        )
        ax.set_xticks([1, 250, 500, 750, 1000])
        ax.set_xticklabels(
            [1, 250, 500, 750, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_ylim(0, 0.26)
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
        ax.set_yticklabels(
            [0, 0.05, 0.1, 0.15, 0.2, 0.25],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Sample size [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Std of $\psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid()
        savefigure.save_as_png(fig, "cumulative_std_vs_sample_size")

    def plot_gradient_cumulative_mean_vs_sample_size(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """Plots the absolute gradient of the cumulative mean of a sample

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        with open("data_representativeness.pkl", "rb") as f:
            [mean_of_cumulative_means, _, _, _, _] = pickle.load(f)

        sample_size = np.arange(1, len(mean_of_cumulative_means) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [
            np.abs(np.diff(mean_of_cumulative_means)[k]) / mean_of_cumulative_means[k]
            for k in range(len(mean_of_cumulative_means) - 1)
        ]
        ax.plot(sample_size[0:-1], gradient, "-k")
        ax.plot(sample_size[0:-1], [1e-2] * len(sample_size[0:-1]), "--r")

        ax.set_xticks([1, 250, 500, 750, 1000])
        ax.set_xticklabels(
            [1, 250, 500, 750, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Sample size [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Mean of $\psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid()
        savefigure.save_as_png(fig, "gradient_cumulative_mean_vs_sample_size")

    def plot_gradient_cumulative_std_vs_size(
        self,
        createfigure,
        savefigure,
        fonts,
    ):
        """
        Plots the absolute gradient of the cumulative std of a sample

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        with open("data_representativeness.pkl", "rb") as f:
            [_, _, cumulative_std, _, _] = pickle.load(f)

        sample_size = np.arange(1, len(cumulative_std) + 1)
        fig = createfigure.rectangle_figure(self.pixels)
        ax = fig.gca()
        gradient = [np.abs(np.diff(cumulative_std)[k]) / cumulative_std[k] for k in range(2, len(cumulative_std) - 1)]
        ax.plot(sample_size[2:-1], gradient, "-k")
        ax.plot(sample_size[2:-1], [1e-2] * len(sample_size[2:-1]), "--r")
        ax.set_xticks([1, 250, 500, 750, 1000])
        ax.set_xticklabels(
            [1, 250, 500, 750, 1000],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 5e-1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Sample size [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"Grad of the Std of $\psi_3$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.grid()
        savefigure.save_as_png(fig, "gradient_cumulative_std_vs_sample_size")


if __name__ == "__main__":

    filename_qMC_Sobol = "dataset_for_metamodel_creation.txt"
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    samplerepresentativeness = SampleRepresentativeness(
        filename_qMC_Sobol, training_amount=0.8, nb_of_shuffled_samples=200, pixels=360
    )

    samplerepresentativeness.plot_PDF_phase3(createfigure, savefigure, fonts)
    
    samplerepresentativeness.compute_means_stds_of_shuffled_samples_and_export_to_pkl()

    samplerepresentativeness.plot_cumulative_mean_vs_sample_size(createfigure, savefigure, fonts)

    samplerepresentativeness.plot_cumulative_std_vs_sample_size(createfigure, savefigure, fonts)

    samplerepresentativeness.plot_gradient_cumulative_mean_vs_sample_size(createfigure, savefigure, fonts)

    samplerepresentativeness.plot_gradient_cumulative_std_vs_size(createfigure, savefigure, fonts)
