# Enhancement of the model of the cellular uptake of nanoparticles with the membrane’s mechano-adaptation : validation with sensitivity analysis

## Associated paper
The present code is the supplemental material associated to the paper ![1](https://doi.org/10.1002/cnm.3598).
Please feel free to contact the authors if you have any question or if you wish to go more into details for some points.

## Dependencies
In order to make sure that you are able to run the code, please install the required versions of the libraries by executing the command bellow in your terminal.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements

```pip3 install -r requirements.txt```


## Tutorial
This repository is divided into 4 folders:
- *model*: contains the code used to compute the total variation of energy of the interface between a circular NP and a membrane by accounting for the mechanical accommodation of the latter. This folder also contains the routine to determine the final wrapping phase of the system. A part of this code was already shared in the repository associated to [2];
- *metamodel*: contains a script to check for the representativeness of the dataset used to create a metamodel, a script to create a Kriging metamodel using the Openturns [3] opensource library, and a routine to validate the metamodel that has just been created;
- *sensitivity_analysis*: contains a script that allows to create samples based on the metamodels that have been created and exported as .pkl files in the metamodel folder. These samples are then used to the  apply sensitivity algorithms. The user can choose among the various sensitivity algorithms provided by Openturns.
- *figures*: contains a utils script to display the graphs and save them as PNG files with consistency.

We recommend the user to follow the following order:
- 1: check for the data representativeness of its sample
- 2: create a metamodel based on this sample, by varying the training amount to determine the metamodel that will provide the best test-prediction performance (validation of the metamodel)
- 3: evaluate the sensitivity of the model to its inputs.

The data that is provided in this repository, in the *dataset_for_metamodel_creation.txt* textfile was obtained by computing the code contained in the *model* folder. To fulfill the criteria of OpenTurns, the document needs to contain only floats, that is why the title of the columns does not appear in this file. They are the following:

| $\overline{\sigma}_r$ | $\overline{\sigma}_{fs}$ | $\overline{\sigma}_{\lambda}$ | $\overline{\gamma}_r$ | $\overline{\gamma}_{fs}$ | $\overline{\gamma}_{\lambda}$ | $\psi_1$ | $\psi_2$ | $\psi_3$ |
|:---------------------:|:------------------------:|:-----------------------------:|:---------------------:|:------------------------:|:-----------------------------:|:--------:|:--------:|:--------:|
|                       |                          |                               |                       |                          |                               |          |          |          |

The notations introduced in the table above are the same as the ones introduced in [1].
If the user wants to use different data, it is recommended that they make a copy of the actual .txt file and that they paste their data in this .txt file. We spotted an error with the type of textfile that is given as an input to create the metamodels (Suggestions to address this issue are more than welcome!)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
A [GPL](https://tldrlegal.com/license/bsd-3-clause-license-(revised)) license is associated to this code, presented in the text file LICENSE.md.

## References
```
[1] @article{
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


