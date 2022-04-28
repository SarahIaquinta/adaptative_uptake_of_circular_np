from io import BytesIO

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
