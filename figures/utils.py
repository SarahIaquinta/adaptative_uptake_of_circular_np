from io import BytesIO

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


class Colors:
    def __init__(self):
        """
        Constructs all the necessary attributes for the DataPreSetting object.

        Parameters:
            ----------
            filename: string
                name of the .txt file from which the data will be extracted
            training_amount : float
                proportion (between 0 and 1) of the initial data used for training
                (the remaining data are used for testing)

        Returns:
            -------
            None
        """

        self.hls_palette = sns.color_palette("hls", 8)

    def hls_palette(self):
        hls_palette = sns.color_palette("hls", 8)
        color_hls_rouge = hls_palette[0]
        color_hls_moutarde = hls_palette[1]
        color_hls_lime = hls_palette[2]
        color_hls_vert = hls_palette[3]
        color_hls_cyan = hls_palette[4]
        color_hls_blue = hls_palette[5]
        color_hls_purple = hls_palette[6]
        color_hls_pink = hls_palette[7]
        return [
            hls_palette,
            color_hls_rouge,
            color_hls_moutarde,
            color_hls_lime,
            color_hls_vert,
            color_hls_cyan,
            color_hls_blue,
            color_hls_purple,
            color_hls_pink,
        ]

    def paired_palette(self):
        paired_palette = sns.color_palette("Paired")
        color_paired_light_blue = paired_palette[0]
        color_paired_dark_blue = paired_palette[1]
        color_paired_light_green = paired_palette[2]
        color_paired_dark_green = paired_palette[3]
        color_paired_light_red = paired_palette[4]
        color_paired_dark_red = paired_palette[5]
        color_paired_light_orange = paired_palette[6]
        color_paired_dark_orange = paired_palette[7]
        color_paired_light_purple = paired_palette[8]
        color_paired_dark_purple = paired_palette[9]
        color_paired_light_marron = paired_palette[10]
        color_paired_dark_marron = paired_palette[11]

        return [
            paired_palette,
            color_paired_light_blue,
            color_paired_dark_blue,
            color_paired_light_green,
            color_paired_dark_green,
            color_paired_light_red,
            color_paired_dark_red,
            color_paired_light_orange,
            color_paired_dark_orange,
            color_paired_light_purple,
            color_paired_dark_purple,
            color_paired_light_marron,
            color_paired_dark_marron,
        ]

    def continuous_flare(self):
        palette = sns.color_palette("flare", as_cmap=True)
        return palette


class Fonts:
    def serif(self):
        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18)
        return font

    def axis_legend_size(self):
        return 24

    def axis_label_size(self):
        return 25

    def fig_title_size(self):
        return 18


class SaveFigure:
    def save_as_png(self, fig, filename):
        png1 = BytesIO()
        fig.savefig(png1, format="png")
        png2 = Image.open(png1)
        png2.save(filename + ".png")

    def save_as_tiff(self, fig, filename):
        png1 = BytesIO()
        fig.savefig(png1, format="png")
        png2 = Image.open(png1)
        png2.save(filename + ".tiff")


class CreateFigure:
    def rectangle_figure(self, pixels):
        fig = plt.figure(figsize=(9, 6), dpi=pixels, constrained_layout=True)
        return fig

    def square_figure(self, pixels):
        fig = plt.figure(figsize=(6, 6), dpi=pixels, constrained_layout=True)
        return fig

    def square_figure_7(self, pixels):
        fig = plt.figure(figsize=(7, 7), dpi=pixels, constrained_layout=True)
        return fig


class XTicks:
    def energy_plots(self):
        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        return xticks


class XTickLabels:
    def energy_plots(self):
        xticklabels = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
        return xticklabels


if __name__ == "__main__":
    colors = Colors()
    [
        hls_palette,
        color_hls_rouge,
        color_hls_moutarde,
        color_hls_lime,
        color_hls_vert,
        color_hls_cyan,
        color_hls_blue,
        color_hls_purple,
        color_hls_pink,
    ] = colors.hls_palette()
    plt.figure()
    x_list = np.linspace(0, 1, 100)
    for i in range(len(hls_palette)):
        y_list = [np.cos(x + i) for x in x_list]
        plt.plot(x_list, y_list, color=hls_palette[i])
    plt.show()
