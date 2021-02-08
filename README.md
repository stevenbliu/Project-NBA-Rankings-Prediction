# DSC180B_Project

**About:**
  This repository contains an implementation of a Graph Convolution Network for node classification on an NBA dataset. The goal being able to classify the ranks of NBA teams using player stats and a graph representation of each matchup between teams in a season. 
  
**Setting Up Docker Image**
  The docker image that was created in order to have an environment able to run this project is found on the repo at https://hub.docker.com/layers/aubarrio/q2checkpoint/latest/images/sha256:cbf58731b23a77bbdbc378934d197965bb54b258c1848c3a68abaff80edb78c5 Or it can be found at aubarrio/q2checkpoint:latest
    
**Model**
  data: The data we use in this project is a compound of multiple webscraped data found on https://www.basketball-reference.com we used stats such as player and team stats, along with team schedules for the season. The seasons for which we collected data range from 2011 to the 2020 season.

**Basic Parameters**
  Since we are in early stages of developing we only have one parameter of choice and that is [train, test]. This is to distinguish the data being input into the model.
    
    train: Parameter train will train the model on all available features (181 different features) 
    test: Parameter test will train the model on 2 features (Rank and Id) which is meant to use as an evaluation of how our data is performing
    
**Examples run.py**
  python run.py test
  python run.py train
  python run.py
  
**Output**
  Direct terminal output outlining the training, validation and test accuracies of our model.
