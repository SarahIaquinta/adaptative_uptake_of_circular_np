


# Enhancement of the model of the cellular uptake of nanoparticles with the membrane’s mechano-adaptation : validation with sensitivity analysis

<div style="text-align: justify">


## Associated paper
The present code is the supplemental material associated to the paper [1]. Please feel free to
contact the authors if you have any question or if you wish to go more into details for some
points.

## Dependencies
In order to make sure that you are able to run the code, please install the required versions of
the libraries by executing the command bellow in your terminal.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements

```pip3 install -r requirements.txt```

# Install the package
Go to the root of the repo and:
``` sh
python -m pip install -e . --user
```

## Tutorial
This repository is divided into 4 folders:
- *model*: contains the code used to compute the total variation of energy of the interface between
  a circular NP and a membrane by accounting for the mechanical accommodation of the latter. This
  folder also contains the routine to determine the final wrapping phase of the system. A part of
  this code was already shared in the
  [repository](https://github.com/SarahIaquinta/uptake_of_random_rigid_elliptic_particle)
  associated to [2];
- *metamodel_implementation*: contains a script to check for the representativeness of the dataset
  used to create a metamodel, a script to create Kriging and PCE metamodels using the Openturns [3]
  opensource library, and a routine to validate the metamodel that has just been created;
- *sensitivity_analysis*: contains a script that allows to create samples based on the Kriging
  metamodels that have been created and exported as .pkl files in the metamodel folder. These
  samples are then used to the  apply sensitivity algorithms. The user can choose among the various
  sensitivity algorithms provided by Openturns. For PCE metamodels, a routine is implemented to
  directly get the Sobol indices from the coefficients of the PCE metamodel. The indices can be
  plotted through plot routines;
- *figures*: contains a utils script to display the graphs and save them as PNG files with
  consistency.

We recommend the user to follow the following order:
- 1: check for the data representativeness of its sample
- 2: create a metamodel based on this sample, by varying the training amount to determine the
  metamodel that will provide the best test-prediction performance (validation of the metamodel)
- 3: evaluate the sensitivity of the model to its inputs.

The data that is provided in this repository, in the *dataset_for_metamodel_creation.txt* textfile
was obtained by computing the code contained in the *model* folder. To fulfill the criteria of
OpenTurns, the document needs to contain only floats, that is why the title of the columns does not
appear in this file. They are the following:



| **<img src="https://render.githubusercontent.com/render/math?math=\overline{\gamma}_r">** | **<img src="https://render.githubusercontent.com/render/math?math=\overline{\gamma}_{fs}">** | **<img src="https://render.githubusercontent.com/render/math?math=\overline{\gamma}_{\lambda}">** | **<img src="https://render.githubusercontent.com/render/math?math=\psi_3">** |
|:-----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
|                                                                                           |                                                                                              |                                                                                                   |

The notations introduced in the table above are the same as the ones introduced in [1]. If the user
wants to use different data, it is recommended that they make a copy of the actual .txt file and
that they paste their data in this .txt file. We spotted an error with the type of textfile that is
given as an input to create the metamodels (Suggestions to address this issue are more than
welcome!)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would
like to change.

Please make sure to update tests as appropriate.

## License
A [GPL](https://tldrlegal.com/license/bsd-3-clause-license-(revised)) license is associated to this
code, presented in the text file LICENSE.md.

</div>

## References
```
[1] @article{
        title={Enhancement of the model of the cellular uptake of nanoparticles
        with the membrane’s mechano-adaptation : validation with
        sensitivity analysis},
        author={Iaquinta, Sarah and Khazaie, Shahram and Ishow, {\'E}l{\'e}na and Blanquart, Christophe and Fr{\'e}our, Sylvain and Jacquemin, Fr{\'e}d{\'e}ric},
        journal={},
        pages={},
        year={},
        publisher={}
        }

[2] @article{
        title={Influence of the mechanical and geometrical parameters on the cellular uptake of nanoparticles: a stochastic approach},
        author={Iaquinta, Sarah and Khazaie, Shahram and Ishow, {\'E}l{\'e}na and Blanquart, Christophe and Fr{\'e}our, Sylvain and Jacquemin, Fr{\'e}d{\'e}ric},
        journal={International Journal for Numerical Methods in Biomedical Engineering},
        pages={e3598},
        year={2022},
        publisher={Wiley Online Library}
        }

[3] @misc{,
        title={Open TURNS: An industrial software for uncertainty quantification in simulation},
        author={Baudin, Micha{\"e}l and Dutfoy, Anne and Iooss, Bertrand and Popelin, Anne-Laure},
        year={2016},
        publisher={Springer International Publishing}
}
