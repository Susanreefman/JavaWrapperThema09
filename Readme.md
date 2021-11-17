# WEKA wrapper

A command-line application around Weka for easy invocation
of Weka by Java processes.

The purpose of this application is to run a program
that accepts a dataset from which the program can
build a model and a datafile with unclassified
instances. The goal is to classify the unclassified
instances based on the built model.

## Requirements
- Dataset (.arff file)
- Dataset with unclassified instances (.arff file)

## Installation and run application
To install by cloning the GitHub repository: \
First clone the repository, open in IDE

Run:
CliOptionsProviderMain.java [options]

Options: \
-i = <input_dataset> - dataset to build model \
-o = <output_dataset> - dataset with unclassified instances \
-h = <help> - help to run application

## In this repository
- Java Classes - running algorithm
- Testdata - see how the application works by running the testdata
    - To run test, type in the command line: \
      CliOptionsProviderMain.java -i data_teds.arff -o unknown_data_teds.arff

----------------------------------------------------------------
Author = Susan Reefman \
Date = 17-11-2021 \



\
h.s.reefman@st.hanze.nl