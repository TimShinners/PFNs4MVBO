# Prior-Data Fitted Networks Can Do Mixed-Variable Bayesian Optimization (PFNs4MVBO)

This repository extends the work done in [PFNs4BO](https://github.com/automl/PFNs4BO) by adapting Prior-Data Fitted Networks (PFNs) to handle Bayesian optimization settings that involve both numerical and categorical variables. 

### Background

PFNs are a type of transformer that have been trained to do Bayesian inference. As input, they take a training data set with a variable number of features and observations, and a test data set containing only inputs. They output predictive distributions over the possible target values. PFNs are trained using synthetic datasets that are sampled from a Prior-Data Generating Model (PDGM), with the choice of PDGM being extremely flexible. 

In [PFNs4BO](https://github.com/automl/PFNs4BO), PFNs demonstrated strong performance as surrogate models for Bayesian optimization.

In our (work)[https://github.com/TimShinners/PFNs4MVBO/blob/main/PFNsCanDoMVBO.pdf], PFNs were trained using a set of Gaussian processes that incorporated different categorical kernels (BODi, Cosmopolitan, CoCaBO). These PFNs performed similarly to their respective Gaussian processes, at a fraction of the computational expense. 

### Use our models











**PFNsCanDoMVBO.pdf** is a pdf of the thesis.

The fully trained PFNs that were used in our experiments are stored in the folder **trained_PFNs**

The scripts **train_COCABO.py**, **train_CASMO.py**, **train_BODI.py**, and **train_MIXED.py** contain code that will define hyperparameters and training settings, then train new PFNs from scratch. This procedure is discussed in Chapter 3 of the thesis.

**evaluate_pfn.sh** contains the code used to evaluate the quality of trained PFNs, prior to experimentation. This procedure is discussed and used in Chapter 4 of the thesis. 

**run_experiments.sh** contains the code used to conduct experiments with the trained PFNs. This code is discussed and used in Chapter 5 of the thesis. 

**generateThesisFigs.ipynb** contains code that was used to generate the figures in the thesis.

The thesis was written in RMarkdown files that were knit using the [thesisdown](https://github.com/ismayc/thesisdown) package. The R files used to produce the PDF of the thesis are stored in **thesis_files**.
