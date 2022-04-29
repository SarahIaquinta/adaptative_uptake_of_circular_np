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
