# 2DeteCTCodes
This is a collection of Python scripts for loading, pre-processing, reconstructing and segmenting X-ray CT projection data of the 2DeteCT data collection as described in

Maximilian B. Kiss, Sophia B. Coban, K. Joost Batenburg, Tristan van Leeuwen, and Felix Lucka "2DeteCT - A large 2D expandable, trainable, experimental Computed Tomography dataset for machine learning", [Sci Data 10, 576 (2023)](https://doi.org/10.1038/s41597-023-02484-6) or [arXiv:2306.05907 (2023)](https://arxiv.org/abs/2306.05907)


* ` Sinogram_production_2DeteCT.ipynb ` was used to produce the sinograms of the 2DeteCT data collection from the raw measurements.
* ` Reconstructions_2DeteCT_2048_allMeth ` encompasses code for various reconstruction methods ('FBP', 'LS', 'NNLS', 'SIRT', 'SART', 'CGLS', 'AGD') to compute reconstructions for the sinogram data of the 2DeteCT data collection.
* ` NesterovGradient.py ` contains the implementation of an accelerated gradient descent (AGD) iterative reconstruction by Henri Der Sarkissian.
* ` Reconstructions_2DeteCT.py ` was used to produce the reference reconstructions of the 2DeteCT data collection from the sinogram data of the first bullet point using the above AGD iterative reconstruction.
* ` Segmentation_2DeteCT.ipynb ` was used to produce the reference segmentations of the 2DeteCT data collection based on the reconstructions of ‘mode 2’ from the bullet point above.
* ` Mode1_settings.csv `, ` Mode2_settings.csv `, ` Mode3_settings.csv ` are machine-readable settings files for the 2DeteCT data collection acquisition.
* ` ReadingSettings_2DeteCT.py ` contains a class for reading in the 2DeteCT acquisition settings from the above mentioned .csv files.

* The complete data collection can be found via the following links: [1-1,000](https://doi.org/10.5281/zenodo.8014757), [1,001-2,000](https://doi.org/10.5281/zenodo.8014765), [2,001-3,000](https://doi.org/10.5281/zenodo.8014786), [3,001-4,000](https://doi.org/10.5281/zenodo.8014828), [4,001-5,000](https://doi.org/10.5281/zenodo.8014873), [5,521-6,370](https://doi.org/10.5281/zenodo.8014906).

* Each slice folder ‘slice00001 - slice05000’ and ‘slice05521 - slice06370’ contains three folders for each mode: ‘mode1’, ‘mode2’, ‘mode3’. In each of these folders there are the sinogram, the dark-field, and the two flat-fields for the raw data archives, or just the reconstructions and for mode2 the additional reference segmentation.

* The corresponding reference reconstructions and segmentations can be found via the following links: [1-1,000](https://doi.org/10.5281/zenodo.8017582), [1,001-2,000](https://doi.org/10.5281/zenodo.8017603), [2,001-3,000](https://doi.org/10.5281/zenodo.8017611), [3,001-4,000](https://doi.org/10.5281/zenodo.8017617), [4,001-5,000](https://doi.org/10.5281/zenodo.8017623), [5,521-6,370](https://doi.org/10.5281/zenodo.8017652).


## Requirements

* Most of the above scripts make use of the [ASTRA toolbox](https://www.astra-toolbox.com/). If you are using conda, this is available through the `astra-toolbox/` channel.

## Contributors

Maximilian Kiss (maximilian.kiss@cwi.nl), CWI, Amsterdam, Henri Der Sarkissian (henri.dersarkissian@gmail.com), Felix Lucka (Felix.Lucka@cwi.nl), CWI, Amsterdam

