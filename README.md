# Prior-Data Fitted Networks Can Do Mixed-Variable Bayesian Optimization

This repo contains much of the code used in my thesis, "Prior-Data Fitted Networks Can Do Mixed-Variable Bayesian Optimization" 

**PFNsCanDoMVBO.pdf** is a pdf of the thesis.

The fully trained PFNs that were used in our experiments are stored in the folder **trained_PFNs**

The scripts **train_COCABO.py**, **train_CASMO.py**, **train_BODI.py**, and **train_MIXED.py** contain code that will define hyperparameters and training settings, then train new PFNs from scratch. This procedure is discussed in Chapter 3 of the thesis.

**evaluate_pfn.sh** contains the code used to evaluate the quality of trained PFNs, prior to experimentation. This procedure is discussed and used in Chapter 4 of the thesis. 

**run_experiments.sh** contains the code used to conduct experiments with the trained PFNs. This code is discussed and used in Chapter 5 of the thesis. 

**generateThesisFigs.ipynb** contains code that was used to generate the figures in the thesis.

The thesis was written in RMarkdown files that were knit using the [thesisdown](https://github.com/ismayc/thesisdown) package. The R files used to produce the PDF of the thesis are stored in **thesis_files**.
