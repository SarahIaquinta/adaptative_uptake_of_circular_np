## These commands should be executed from the root of the repository (one level above np_uptake folder)

## Plot Figures 2.a. (Energy vs wrapping degree) and 2.b. (NP-membrane system)
# The following command generated both Figures 2.a. and 2.b. 
# More especially, the function plot_energy displays the evolution of the total
# potential energy in terms of the wrapping degree f (Figure 2.a.) and 
# the function plot_np_membrane_wrapping displays the NP-membrane system for 
# a given wrapping degree f (Figure 2.b.). The value of f is specified in line 834. 
# By default, it is the value of the wrapping degree at equilibrium.
echo "Plot Figures 2.a"
python -W ignore np_uptake/model/cellular_uptake_rigid_particle.py     --r_bar 0.3     --particle_perimeter 6.28 --gamma_bar_r 1 --sigma_bar_0 2 --gamma_bar_0 6

## Plot Figures 5 (Evolution of gamma_bar in terms of the wrapping degree f)
echo "Plot Figures 5"
python -W ignore np_uptake/model/system_definition.py     --r_bar 0.3     --particle_perimeter 6.28 --gamma_bar_r 2 --sigma_bar_0 2 --gamma_bar_0 1 --gamma_bar_fs 0 --gamma_bar_lambda 50

## Plot Figure 7.a. (Phase diagram)
# Obs: It takes approximately 1 hour without parallelization to generate the data necessary for plotting the phase diagram. 
# As such, the dataset is first generated and then stored in a text file to avoid generating the data again for plotting. 
# The data used to plot Figure 7.a. is provided as "data_for_phase_diagram_1.txt".
# Note that in this case, the "1" after the underscore at the end of the filename stands for the NP's aspect ratio r_bar.
# The function necessary for generating this data (generate_phase_diagram_dataset) is coded in the file np_uptake/model/phase_diagrams.py 
#but it not called (line 126 is commented)
echo "Plot Figures 7.a"
python -W ignore np_uptake/model/phase_diagrams.py 

## Plot Figure 7.b. (Phase proportions in terms of r_bar)
# Obs: It takes approximately 1 week without parallelization to generate the data necessary for plotting the phase diagram. 
# As such, the dataset is first generated and then stored in a text file to avoid generating the data again for plotting. 
# The data used to plot Figure 7.b. is provided as "data_for_phase_proportion_vs_r.txt".
# The function necessary for generating this data is coded in the file np_uptake/model/phase_proportions.py but it not called (line 111 is commented)
echo "Plot Figures 7.b"
python -W ignore np_uptake/model/phase_proportions.py 

### For the next figures, the way of plotting them is the same, the only difference is the value of the input parameter --r_bar used. For Figure 8 to 10, r_bar = 1 (circular NP) and from Figure 11 to 13 the value of r_bar varies within its domain of definition.
## Plot Figures 8a, 8b and 8c and 11a, 11b and 11c
echo "Plot Figures 8.a,bc, 11a,c"
python -W ignore np_uptake/metamodel_implementation/data_representativeness.py

## Plot Figures 9a and 9b and 12a and 12b
# The following commands execute the routines that create the metamodels, store them
# into .pkl files, and then uses these .pkl files for the validation
# routine of the metamodels.
# It is possible to specify the training_amount of both Kriging and PCE metamodels, 
# and also the degrees to be tested for PCE. 
echo "Plot Figures 9a,b, 12a,b"
python -W ignore np_uptake/metamodel_implementation/metamodel_creation.py
python -W ignore np_uptake/metamodel_implementation/metamodel_validation.py

## Plot Figures 10a and 10b and 13a and 13b
# The following command calls a routine for computing the Sobol sensitivity indices, 
# running a convergence study in terms of the size of the dataset, 
# and displaying the Sobol indices into figures.
echo "Plot Figures 10a,b, 13a,b"
python -W ignore np_uptake/sensitivity_analysis/sensitivity_analysis.py