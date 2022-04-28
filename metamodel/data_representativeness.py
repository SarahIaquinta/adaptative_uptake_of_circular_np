import os


path = os.path.dirname(os.path.abspath(__file__))



import pickle

import numpy as np
import openturns as ot

ot.Log.Show(ot.Log.NONE)

from figures.utils import CreateFigure, Fonts, SaveFigure


class SampleRepresentativeness:
    def __init__(self, filename, training_amount, nb_of_shuffled_samples, pixels):
        """
        Constructs all the necessary attributes for the DataPreSetting object.

        Parameters:
            ----------
            filename: string
                name of the .txt file from which the data will be extracted
            training_amount: float
                proportion (between 0 and 1) of the initial data used for training (the remaining
                data are used for testing)
            nb_of_shuffled_samples: float
                number of shuffled samples of the dataset, used to evaluate the standard
                deviation of the cumulative mean of a sample
            pixels: string
                number of points per pixel in the figures
                Recommended: 360


        Returns:
            -------
            None
        """

        self.filename = path + "/" + filename
        self.training_amount = training_amount
        self.pixels = pixels
        self.nb_of_shuffled_samples = nb_of_shuffled_samples

    def generate_shuffled_samples(self):
        """
        Shuffles the dataset output (phase 3) self.nb_of_shuffled_samples times

        Parameters:
            ----------
            None

        Returns:
            -------
            all_shuffled_phase3: array of shape ((len(sample) , self.nb_of_shuffled_samples))
                self.nb_of_shuffled_samples times shuffled phase 3

        """
        sample = ot.Sample.ImportFromTextFile(self.filename, "\t", 0)
        phase3 = sample[:, -1]
        all_shuffled_phase3 = np.zeros((len(phase3), self.nb_of_shuffled_samples))
        for i in range(self.nb_of_shuffled_samples):
            np.random.shuffle(phase3)
            for k in range(len(phase3)):
                all_shuffled_phase3[k, i] = phase3[k, 0]
        return all_shuffled_phase3

    def compute_cumulative_mean_std(self, vector):
        """
        Computes the cumulative mean and standard deviation (std) of a vector

        Parameters:
            ----------
            vector: array
                vector of which the cumulative mean and std will be computed

        Returns:
            -------
            cumulative_mean: array, same shape as vector
                cumulative mean of the vector
            cumulative_std: array, same shape as vector
                cumulative std of the vector

        """
        cumulative_mean = np.zeros(len(vector))
        cumulative_std = np.zeros_like(cumulative_mean)
        for i in range(len(vector)):
            cumulative_mean[i] = np.mean(vector[0:i])
            cumulative_std[i] = np.std(vector[0:i])
        return cumulative_mean, cumulative_std

    def compute_means_stds_of_shuffled_samples_and_export_to_pkl(self):
        """
        Computes the cumulative mean and standard deviation (std) for the
            self.nb_of_shuffled_samples shuffled samples that
            have been generated
        Exports them into a .pkl file

        Parameters:
            ----------
            None

        Returns:
            -------
            None

        Exports:
            -------
            cumulative_mean: array
                cumulative mean of one shuffled sample
            std_of_cumulative_means: array
                std of the cumulative_mean of all the shuffled samples
            cumulative_std: array
                cumulative std of one shuffled sample
            std_of_cumulative_stds: array
                std of the cumulative_std of all the shuffled samples
            all_shuffled_phase3: array
                shuffled samples used in this method.
                output of the self.generate_shuffled_samples() method

            These objects are exported in a .pkl file named "data_representativeness.pkl"

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
        """
        Plots the cumulative mean of a sample with the
            std (computed from the self.nb_of_shuffled_samples shuffled
                samples)

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
        ax.set_ylim(-0.02, 0.55)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5],
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
        """
        Plots the cumulative std of a sample with the
            std (computed from the self.nb_of_shuffled_samples shuffled
                samples)

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
        ax.set_ylim(0, 0.2)
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
        ax.set_yticklabels(
            [0, 0.05, 0.1, 0.15, 0.2],
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
        """
        Plots the absolute gradient of the cumulative mean of a sample

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

    samplerepresentativeness.compute_means_stds_of_shuffled_samples_and_export_to_pkl()

    samplerepresentativeness.plot_cumulative_mean_vs_sample_size(createfigure, savefigure, fonts)

    samplerepresentativeness.plot_cumulative_std_vs_sample_size(createfigure, savefigure, fonts)

    samplerepresentativeness.plot_gradient_cumulative_mean_vs_sample_size(createfigure, savefigure, fonts)

    samplerepresentativeness.plot_gradient_cumulative_std_vs_size(createfigure, savefigure, fonts)
