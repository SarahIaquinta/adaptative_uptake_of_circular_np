
import seaborn as sns
import numpy as np
import itertools
import np_uptake.model.cellular_uptake_rigid_particle as cellupt
import np_uptake.figures.utils as fiu
import np_uptake.model.system_definition as sysdef

def generate_phase_diagram_dataset(args, r_bar):
    particle = sysdef.ParticleGeometry(r_bar=r_bar, particle_perimeter=args.particle_perimeter, sampling_points_circle=300)
    membrane = sysdef.MembraneGeometry(particle, sampling_points_membrane=100)
    wrapping = sysdef.Wrapping(wrapping_list=np.arange(0.03, 0.97, 0.003125))
    energy_computation = cellupt.EnergyComputation()
    gamma_bar_0_list = np.arange(1, 8.5, 0.5)
    sigma_bar_0_list = np.arange(0.5, 5.75, 0.25)
    generator = list(
    itertools.product(
        gamma_bar_0_list,
        sigma_bar_0_list)
)
    f = open('data_for_phase_diagram_'+ str(r_bar) +  '.txt', "w")
    f.write("r_bar \t gA \t gD \t gS \t sigma_bar_0 \t gamma_bar_0 \t phase\n")
    for i in range(len(generator)):
        tuple = generator[i]
        gamma_bar_0, sigma_bar_0 = tuple
        mechanics = sysdef.MechanicalProperties_Adaptation(
            testcase="testcase",
            gamma_bar_r=args.gamma_bar_r,
            gamma_bar_fs=args.gamma_bar_fs,
            gamma_bar_lambda=args.gamma_bar_lambda,
            gamma_bar_0=gamma_bar_0,
            sigma_bar=sigma_bar_0,
        )
        f_eq, wrapping_phase_number, wrapping_phase, energy_list, time_list = cellupt.identify_wrapping_phase(
            particle, mechanics, membrane, wrapping, energy_computation
        )
        f.write(str(args.r_bar) +"\t" + str(args.gamma_bar_r) +"\t" + str(args.gamma_bar_fs) + "\t" + str(args.gamma_bar_lambda) + "\t" + str(sigma_bar_0) +"\t" + str(gamma_bar_0) + "\t" + str(wrapping_phase_number) + "\n")

def read_data_for_phasediagram_from_datafile(filename):
    with open(filename) as f:
        next(f)
        # line = f.readline()
        gamma_bar_0_list_phase1 = []
        gamma_bar_0_list_phase2 = []
        gamma_bar_0_list_phase3 = []
        sigma_bar_0_list_phase1 = []
        sigma_bar_0_list_phase2 = []
        sigma_bar_0_list_phase3 = [] 
        for line in f:
            print(line, end='')
            values = line.split()
            [r_bar, gamma_bar_r, gamma_bar_fs, gamma_bar_lambda, sigma_bar_0, gamma_bar_0, phase] = [float(s) for s in values]
            if phase == 1:
                gamma_bar_0_list_phase1.append(gamma_bar_0)
                sigma_bar_0_list_phase1.append(sigma_bar_0)
            elif phase == 2:
                gamma_bar_0_list_phase2.append(gamma_bar_0)
                sigma_bar_0_list_phase2.append(sigma_bar_0)
            elif phase == 3:
                gamma_bar_0_list_phase3.append(gamma_bar_0)
                sigma_bar_0_list_phase3.append(sigma_bar_0)
    return gamma_bar_0_list_phase1, gamma_bar_0_list_phase2, gamma_bar_0_list_phase3, sigma_bar_0_list_phase1, sigma_bar_0_list_phase2, sigma_bar_0_list_phase3


def plot_phase_diagram(filename, createfigure, fonts, xticks, xticklabels, savefigure):
    gamma_bar_0_list_phase1, gamma_bar_0_list_phase2, gamma_bar_0_list_phase3, sigma_bar_0_list_phase1, sigma_bar_0_list_phase2, sigma_bar_0_list_phase3 = read_data_for_phasediagram_from_datafile(filename)
    fig = createfigure.square_figure_7(pixels=360)
    ax = fig.gca()
    paired_palette = sns.color_palette("Paired")
    
    ax.plot(
        sigma_bar_0_list_phase1,
        gamma_bar_0_list_phase1,
        "o",
        markersize=5,
        color=paired_palette[8],
        label=r"no wrapping",
    )
    ax.plot(
        sigma_bar_0_list_phase2,
        gamma_bar_0_list_phase2,
        "o",
        markersize=5,
        color=paired_palette[6],
        label=r"partial wrapping",
    )
    ax.plot(
        sigma_bar_0_list_phase3,
        gamma_bar_0_list_phase3,
        "o",
        markersize=5,
        color=paired_palette[2],
        label=r"full wrapping",
    )    
    ax.set_xticks([0.5, 1, 1.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    ax.set_xticklabels(
        ["0.5", "1", "1.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
    ax.set_yticklabels(
        ["1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6", "6.5", "7", "7.5", "8"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlabel(r"$\overline{\sigma}$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"$\overline{\gamma}_0$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.9)
    savefigure.save_as_png(fig, "phasediagram")    

        
if __name__ == "__main__":
    args = cellupt.parse_arguments()
    createfigure = fiu.CreateFigure()
    fonts = fiu.Fonts()
    savefigure = fiu.SaveFigure()
    xticks = fiu.XTicks()
    xticklabels = fiu.XTickLabels()
    # r_bar=1
    # generate_phase_diagram_dataset(args, r_bar)
    plot_phase_diagram("data_for_phase_diagram_1.txt", createfigure, fonts, xticks, xticklabels, savefigure)
