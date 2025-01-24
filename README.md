This repository contains the CGP (Conditional Gaussian Process) algorithm for time series. See the article "Fast analytical method to integrate multivariate Gaussians over hidden variables", see https://hal.science/hal-04692487v1/document for more details. This tool aims to facilitate the implementation of probabilistic models for time series.

## Abstract

Many biological and physical datasets can be modeled as Gaussian processes, whose parameters can be estimated using maximum likelihood estimation. The likelihood can be expressed as a product of conditional probabilities, where each term provides information about the model's behavior and dynamics. Here, we present a versatile framework for retrieving the physical properties of these models using products of univariate Gaussians that depend on a set of observed and hidden variables. To do so, our model uses a recurrence formula to integrate over the hidden variables, enabling fast and accurate computing.
This framework is particularly effective for characterizing recurrent Gaussian processes. In the last section, we show the use of our framework via three microscopy applications: determining the type of motion of a particle, modeling time-dependent changes in FRET efficiency, and detecting confinement between two particles.

## Installation

Download the github repository, decompress to the destination of your choice and set the working directory to the folder called `CGP`. Run the file `Turorial_Anomalous_Tracking.py` to verify that the framework is working.

## Principle

This tool can be used to model recurrent gaussians processes with observed variables and hidden variables. In many cases, as exemplified in our article (See the link above), we consider a temporal sequence of data where we follow the evolution of observed variables $x_i$. In many physical processes, the same forces are applied at every time point resulting in a recurrent process that typically also depend on hidden variables that also change with time $y_i$. If at a given time step $i$ of a recurrent process, the joint probability of the observed variables $x_i$ and the hidden variables $y_i$ can be expressed as a product of Gaussians that depend on linear combinations of the current and previous observed and hidden variables ($x_{i-1}$ and $y_{i-1}$), one can use the CGP framework to compute the probability of the observed data by integration of the joint probability over the hidden variables.

## Determining the process

To apply our model, we recommand the user to first identify the underlying variables that have degrees of freedom with regard to the observed variables and to the other hidden variables. Then, the user should try to express the probability of the current observed and hidden variables as a product of univariate Gaussians that depend on linear combinations the previous and current variables of the model, similarly to the methodology used in our article. If this is done correctly, the number of Gaussians should be equal to the sum of the number of hidden and observed variables (for one time step). NB: some processes cannot be expressed as such either because the random process is not Gaussian or because the process require non-linear combinations the previous and current variables of the model. If non-linear combinations are required, it might be possible to simplify the problem to find a good linear approximation as in our FRET example.
It is also important to identify how to initialize the hidden variables.

## Defining the constraint function
Once the recursive relations are identified, the user must translate them into a constraint function that will specify the relations between the variables and the model parameters.








