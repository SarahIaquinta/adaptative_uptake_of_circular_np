from io import BytesIO

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Fonts:
    """A class that defines the features of the fonts that will be used in the figures

    Attributes:
    ----------
    None

    Methods:
    -------
    serif(self):
        Introduces the serif font
    """

    def serif(self):
        """Introduces the serif font

        Parameters:
        ----------
        None

        Returns:
        -------
        font: font_manager object
            Serif font, with its features
        """

        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18)
        return font

    def serif_3horizontal(self):
        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=40)
        return font

    def axis_legend_size(self):
        """Defines the size of the text in the axis legend of the figures

        Parameters:
        ----------
        None

        Returns:
        -------
        size_axis_legend: float
            Size of the text in the axis legend of the figures
        """

        size_axis_legend = 24
        return size_axis_legend

    def axis_label_size(self):
        """Defines the size of the text in the axis label of the figures

        Parameters:
        ----------
        None

        Returns:
        -------
        size_axis_label: float
            Size of the text in the axis label of the figures
        """
        size_axis_label = 25
        return size_axis_label

    def fig_title_size(self):
        """Defines the size of the title the figures

        Parameters:
        ----------
        None

        Returns:
        -------
        size_title: float
            Size of the text in the axis legend of the figures
        """

        size_title = 18
        return size_title


class SaveFigure:
    """A class which contains methods to export and save figures as image (.png or .tiff) files

    Attributes:
    ----------
    None

    Methods:
    -------
    save_as_png(self, fig, filename):
        Saves the figure as a .png file
    save_as_tiff(self, fig, filename):
        Saves the figure as a .tiff file
    """

    def save_as_png(self, fig, filename):
        """Saves the figure as a .png file

        Parameters:
        ----------
        fig: Figure object
            Figure that is to be saved as an image
        filename: str
            Name of the .png file under which the image must be saved

        Returns:
        -------
        None
        """

        png1 = BytesIO()
        fig.savefig(png1, format="png")
        png2 = Image.open(png1)
        png2.save(filename + ".png")

    def save_as_tiff(self, fig, filename):
        """Saves the figure as a .tiff file

        Parameters:
        ----------
        fig: Figure object
            Figure that is to be saved as an image
        filename: str
            Name of the .tiff file under which the image must be saved

        Returns:
        -------
        None
        """

        png1 = BytesIO()
        fig.savefig(png1, format="png")
        png2 = Image.open(png1)
        png2.save(filename + ".tiff")


class CreateFigure:
    """A class which contains the dimensions used to create the figures

    Attributes:
    ----------
    None

    Methods:
    -------
    rectangle_figure(self, pixels):
        Creates a rectangle figure of size 9x6
    square_figure(self, pixels):
        Creates a square figure of size 6x6
    square_figure_7(self, pixels):
        Creates a square figure of size 7x7
    """

    def rectangle_figure(self, pixels):
        """Creates a rectangle figure of size 9x6

        Parameters:
        ----------
        pixels: str
            Number of points per pixel in the figures Recommended: 360

        Returns:
        -------
        fig: Figure object
            Figure of dimension 9x6
        """

        fig = plt.figure(figsize=(9, 6), dpi=pixels, constrained_layout=True)
        return fig

    def square_figure(self, pixels):
        """Creates a square figure of size 6x6

        Parameters:
        ----------
        pixels: str
            Number of points per pixel in the figures Recommended: 360

        Returns:
        -------
        fig: Figure object
            Figure of dimension 6x6
        """

        fig = plt.figure(figsize=(6, 6), dpi=pixels, constrained_layout=True)
        return fig

    def square_figure_7(self, pixels):
        """Creates a square figure of size 7x7

        Parameters:
        ----------
        pixels: str
            Number of points per pixel in the figures Recommended: 360

        Returns:
        -------
        fig: Figure object
            Figure of dimension 7x7
        """

        fig = plt.figure(figsize=(7, 7), dpi=pixels, constrained_layout=True)
        return fig


class XTicks:
    """A class which defines the position of the labels that must appear in the abscises of the
    plots

    Attributes:
    ----------
    None

    Methods:
    -------
    energy_plots(self):
        Defines the position where the labels that must appear in the abscises of the plots of
        energy
    """

    def energy_plots(self):
        """Defines the position where the labels that must appear in the abscises of the plots of
        energy

        Parameters:
        ----------
        None

        Returns:
        -------
        xticks: list of floats
            list of the position of the labels of the x axis
        """

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        return xticks


class XTickLabels:
    """A class which defines the labels in the abscises of the plots

    Attributes:
    ----------
    None

    Methods:
    -------
    energy_plots(self):
        Defines the labels that must appear in the abscises of the plots of energy
    """

    def energy_plots(self):
        """Defines the labels that must appear in the abscises of the plots of energy

        Parameters:
        ----------
        None

        Returns:
        -------
        xticks: list of strings
            list of the labels for the x axis
        """

        xticklabels = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
        return xticklabels


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
        