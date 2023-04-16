# Capstone Project: German Traffic Sign Recognition Benchmark
# AWS Machine Learning Engineer Nanodegree

## Problem statement and motivation
This repository includes the code and the report for my Capstone Project in the AWS Machine Learning Engineer Nanodegree. 
My task was to create a end-to-end solution for the German Traffic Sign Recognition Benchmark. The target in it is to 
be able to recognize certain traffic sign groups when given an input image. I also extended the problem by also creating a solution
for the task of actually recognizing each traffic signs exact class.

I chose this topic as I'm very interested in the field of autonomous driving and it represents a very real problem in this field.

## Used libraries
My solution makes use of all the libraries which are available due to the use of Amazon Sagemaker Studio.
The implementation of my neural network was done in PyTorch.
The solution also deploys each model to an Amazon endpoint and uses the corresponding libraries.

## Files in this repository
- train_and_deploy.ipynb: This Jupyter notebook implements all the logic to run my solution and combines the necessary python scripts. It does the preprocessing, training and then deploys the models.
- train_model.py: This python script implements the basic training of neural network.
- hpo_combined.py: This file is used to run the hyperparameter tuning for the neural network with the combined traffic sign groups.
- hpo_distinct.py: This file is used to run the hyperparameter tuning for the neural network with the distinct traffic sign classes.
- endpoint_inference_combined.py: This file implements the logic for the endpoint with the model for the combined traffic sign groups.
- endpoint_inference_distinct.py: This file implements the logic for the endpoint with the model for the distinct traffic sign classes.
- proposal.pdf: The proposal for my capstone project.
- report.pdf: My capstone project report.

## How to run
Copy all the included files into a Sagemaker studio instance. Download the dataset as mentioned in the corresponding section and extract it in
Sagemaker Studio into a folder called TrafficSignImages (needs to be created first). Open the Jupyter notebook and chose for the Kernel the Data Science Image and the Python3 Kernel.
My used instance was ml.t3.medium. Then simply run all the cells.

## Download dataset
The dataset can be downloaded from the following website:

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

A Kaggle account is required for the download.

## Results
The final solution model for the combined approach has an accuracy of 96.4%. While that is way higher than the baseline model with an accuracy of 59.5%, it falls a few percent points behind the benchmark with an accuracy of 98.98%.

The final solution model for the distinct approach has an accuracy of 62.6%. It is therefore lower due to the higher complexity but definitely beats my baseline with an accuracy of 21.4%.

While it does not beat the benchmark, it gets very close and shows just what is possible with a few optimizations. 