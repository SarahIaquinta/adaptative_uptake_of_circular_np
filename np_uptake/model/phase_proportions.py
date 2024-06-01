import seaborn as sns
import numpy as np
import itertools
import np_uptake.model.cellular_uptake_rigid_particle as cellupt
import np_uptake.figures.utils as fiu
import np_uptake.model.system_definition as sysdef
import np_uptake.model.phase_diagrams as phdiag
import tikzplotlib
from np_uptake.figures.utils import tikzplotlib_fix_ncols
from pathlib import Path

def generate_phase_proportion_dataset(args):
    r_list_horizontal1 = list(np.arange(1, 3.02, 0.02))
    r_list_horizontal2 = list(np.linspace(3, 6, 11))
    r_list_horizontal = r_list_horizontal1 + r_list_horizontal2
    r_list_vertical = [np.round(1 / r, 3) for r in r_list_horizontal]
    r_list = r_list_horizontal + r_list_vertical
    f = open('data_for_phase_proportion_vs_r.txt', "w")
    f.write("gA \t gD \t gS \t rbar \t prop phase1 \t prop phase2 \t prop phase3 \n")        
    for i in range(len(r_list)):
        r_bar = r_list[i]
        phdiag.generate_phase_diagram_dataset(args, r_bar)
        gamma_bar_0_list_phase1, gamma_bar_0_list_phase2, gamma_bar_0_list_phase3, sigma_bar_0_list_phase1, sigma_bar_0_list_phase2, sigma_bar_0_list_phase3 = phdiag.read_data_for_phasediagram_from_datafile('data_for_phase_diagram_'+ str(r_bar) +  '.txt')
        proportion_phase1 = len(gamma_bar_0_list_phase1) / (len(gamma_bar_0_list_phase1) + len(gamma_bar_0_list_phase2) + len(gamma_bar_0_list_phase3))
        proportion_phase2 = len(gamma_bar_0_list_phase1) / (len(gamma_bar_0_list_phase1) + len(gamma_bar_0_list_phase2) + len(gamma_bar_0_list_phase3))
        proportion_phase3 = 1 - (proportion_phase1 + proportion_phase2)
        f.write(str(args.gamma_bar_r) + "\t" + str(args.gamma_bar_fs) + "\t" + str(args.gamma_bar_lambda) + "\t" + str(r_bar) + "\t" + str(proportion_phase1) + "\t" + str(proportion_phase2)+ "\t" + str(proportion_phase3) + "\n")

def read_data_for_phaseproportions_from_datafile(filename):
    with open(filename) as f:
        next(f)
        num_lines = 0
        for line in f:
            num_lines += 1
    r_bar_vertical_list = []
    proportion_phase1_vertical_list = []
    proportion_phase2_vertical_list = []
    proportion_phase3_vertical_list = []
    r_bar_horizontal_list = []
    proportion_phase1_horizontal_list = []
    proportion_phase2_horizontal_list = []
    proportion_phase3_horizontal_list = []
    with open(filename) as f:
        next(f)
        for line in f:
            values = line.split()
            [gamma_bar_r, gamma_bar_fs, gamma_bar_lambda, r_bar, proportion_phase1, proportion_phase2, proportion_phase3] = [float(s) for s in values]
            if r_bar < 1:
                r_bar_vertical_list.append(r_bar)
                proportion_phase1_vertical_list.append(proportion_phase1)
                proportion_phase2_vertical_list.append(proportion_phase2)
                proportion_phase3_vertical_list.append(proportion_phase3)
            else:
                r_bar_horizontal_list.append(r_bar)
                proportion_phase1_horizontal_list.append(proportion_phase1)
                proportion_phase2_horizontal_list.append(proportion_phase2)
                proportion_phase3_horizontal_list.append(proportion_phase3) 
    phase_proportions_vs_rbar_vertical = np.zeros((len(r_bar_vertical_list), 4))       
    for i in range(len(r_bar_vertical_list)):
        phase_proportions_vs_rbar_vertical[i, 0] = r_bar_vertical_list[i]
        phase_proportions_vs_rbar_vertical[i, 1] = proportion_phase1_vertical_list[i]
        phase_proportions_vs_rbar_vertical[i, 2] = proportion_phase2_vertical_list[i]
        phase_proportions_vs_rbar_vertical[i, 3] = proportion_phase3_vertical_list[i]
    phase_proportions_vs_rbar_horizontal = np.zeros((len(r_bar_horizontal_list), 4))       
    for i in range(len(r_bar_horizontal_list)):
        phase_proportions_vs_rbar_horizontal[i, 0] = r_bar_horizontal_list[i]
        phase_proportions_vs_rbar_horizontal[i, 1] = proportion_phase1_horizontal_list[i]
        phase_proportions_vs_rbar_horizontal[i, 2] = proportion_phase2_horizontal_list[i]
        phase_proportions_vs_rbar_horizontal[i, 3] = proportion_phase3_horizontal_list[i]
    return phase_proportions_vs_rbar_vertical, phase_proportions_vs_rbar_horizontal

def plot_phase_proportions_vs_rbar(filename, createfigure, fonts, savefigure):
    phase_proportions_vs_rbar_vertical, phase_proportions_vs_rbar_horizontal = read_data_for_phaseproportions_from_datafile(filename)
    r_list_horizontal = phase_proportions_vs_rbar_horizontal[:, 0]
    r_list_vertical_to_horizontal = [1/r for r in phase_proportions_vs_rbar_vertical[:, 0]]
    fig = createfigure.rectangle_figure(pixels=360)
    ax = fig.gca()
    kwargs = {"linewidth": 4.5}
    paired_palette = sns.color_palette("Paired")
    ax.plot(r_list_horizontal, phase_proportions_vs_rbar_horizontal[:, 1], 'o', color=paired_palette[8], label=r"$\psi_1$, horizontal NPs", **kwargs)
    ax.plot(r_list_horizontal, phase_proportions_vs_rbar_horizontal[:, 2], 'o', color=paired_palette[6], label=r"$\psi_2$, horizontal NPs", **kwargs)
    ax.plot(r_list_horizontal, phase_proportions_vs_rbar_horizontal[:, 3], 'o', color=paired_palette[2], label=r"$\psi_3$, horizontal NPs", **kwargs)
    ax.plot(r_list_vertical_to_horizontal, phase_proportions_vs_rbar_vertical[:, 1], 'o', color=paired_palette[9], label=r"$\psi_1$, vertical NPs", **kwargs)
    ax.plot(r_list_vertical_to_horizontal, phase_proportions_vs_rbar_vertical[:, 2], 'o', color=paired_palette[7], label=r"$\psi_2$, vertical NPs", **kwargs)
    ax.plot(r_list_vertical_to_horizontal, phase_proportions_vs_rbar_vertical[:, 3], 'o', color=paired_palette[3], label=r"$\psi_3$, vertical NPs", **kwargs)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "2", "3", "4", "5"], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], font=fonts.serif(), fontsize=fonts.axis_legend_size())
    ax.set_xlabel(r"$\overline{r}$ for horizontal NPs  [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Phase proportion  [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    secax = ax.secondary_xaxis("top")
    secax.set_xticks([1, 2, 3, 4, 5])
    secax.set_xticklabels(
        ["1", "1/2", "1/3", "1/4", "1/5"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    secax.set_xlabel(r"$\overline{r}$ for vertical NPs  [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="center right", framealpha=0.9)
    savefigure.save_as_png(fig, "phaseproportion_vs_r")
    tikzplotlib_fix_ncols(fig)
    current_path = Path.cwd()
    tikzplotlib.save(current_path/"phaseproportion_vs_r.tex")
    
if __name__ == "__main__":
    args = cellupt.parse_arguments()
    createfigure = fiu.CreateFigure()
    fonts = fiu.Fonts()
    savefigure = fiu.SaveFigure()
    xticks = fiu.XTicks()
    xticklabels = fiu.XTickLabels()
    
    # generate_phase_proportion_dataset(args)
    plot_phase_proportions_vs_rbar("data_for_phase_proportion_vs_r.txt", createfigure, fonts, savefigure)
    
        
