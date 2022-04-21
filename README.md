# Enhancement of the model of the cellular uptake of nanoparticles with the membrane’s mechano-adaptation : validation with sensitivity analysis

## Associated paper
The present code is the supplemental material associated to the paper ![1](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle/blob/main/Cellular_uptake_of_rigid_elliptic_random_nanoparticles.pdf). 

## Dependencies
In order to make sure that you are able to run the code, please install the required versions of the libraries by executing the command bellow in your terminal.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements

```pip3 install -r requirements.txt```

## Figures

![coordinates](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle/blob/main/figures/coordinates.png)
Figure 1: Definition of the coordinates system

![psi1](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle/blob/main/figures/psi1_gif.gif)
Figure 2: Definition of psi 1 angle

![psi3](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle/blob/main/figures/psi3_gif.gif)
Figure 3: Definition of psi 3 angle

![beta](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle/blob/main/figures/beta_angles.png)
Figure 4: Definition of beta angles

![theta](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle/blob/main/figures/theta_angles.png)
Figure 5: Definition of theta angles

![delta](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle/blob/main/figures/delta_angles.png)
Figure 5: Definition of delta angle

## Tutorial
This repository is divided into 4 folders:
- *model*: contains the code used to compute the total variation of energy of the interface between a circular NP and a membrane by accounting for the mechanical accommodation of the latter. This folder also contains the routine to determine the final wrapping phase of the system. A part of this code was already shared in the repository associated to [2];
- *metamodel*: contains a script to check for the representativeness of the dataset used to create a metamodel, a script to create a Kriging metamodel using the Openturns [3] opensource library, and a routine to validate the metamodel that has just been created;
- *sensitivity_analysis*: contains a script that allows to create samples based on the metamodels that have been created and exported as .pkl files in the metamodel folder. These samples are then used to the  apply sensitivity algorithms. The user can choose among the various sensitivity algorithms provided by Openturns. 
- *figures*: contains a utils script to display the graphs and save them as PNG files with consistency.



Remark: Depending on your python version, you might use a different command than "python", as "py", "py3" or "python3" for instance. 

It is also possible to run the code from any Python development environment. It will run the code written in the main section.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
A [GPL](https://tldrlegal.com/license/bsd-3-clause-license-(revised)) license is associated to this code, presented in the text file LICENSE.md.

## References
```
[1] @article{,
        title={Enhancement of the model of the cellular uptake of nanoparticles
        with the membrane’s mechano-adaptation : validation with
        sensitivity analysis},
        author={Iaquinta, Sarah and Khazaie, Shahram and Ishow, {\'E}l{\'e}na and Blanquart, Christophe and Fr{\'e}our, Sylvain and Jacquemin, Fr{\'e}d{\'e}ric},
        journal={International Journal for Numerical Methods in Biomedical Engineering},
        pages={e3598},
        year={2022},
        publisher={Wiley Online Library}
        }

[2] @article{
        title={Influence of the mechanical and geometrical parameters on the cellular uptake of nanoparticles: a stochastic approach},
        author={Iaquinta, Sarah and Khazaie, Shahram and Ishow, {\'E}l{\'e}na and Blanquart, Christophe and Fr{\'e}our, Sylvain and Jacquemin, Fr{\'e}d{\'e}ric},
        journal={},
        pages={},
        year={},
        publisher={}
        }

[3] @misc{,
        title={Open TURNS: An industrial software for uncertainty quantification in simulation},
        author={Baudin, Micha{\"e}l and Dutfoy, Anne and Iooss, Bertrand and Popelin, Anne-Laure},
        year={2016},
        publisher={Springer International Publishing}
} 

```
