# Food-AI-Recommender-System
### Table of Contents

1. [Summary and Motivation](#installation)
2. [File Descriptions](#files)
3. [Results](#results)
4. [Licensing, Authors](#licensing)
5. [Instructions](#instructions)

## Summary and Motivation <a name="installation"></a>
This is made by Food 101 dataset in which there are 101 categories of food are avaliable This dataset contain 101000 image including train and test You can download from here https://www.vision.ee.ethz.ch/datasets_extra/food-101/ I make model using this dataset with 80% accuracy on validation data and 72% accuracy on test data  using Inception v3 neural network made by Google
This project cosist of two parts in First part We detect the food than we recommend food on the basis of Height,Weight and Gender using calorires per weight of food
In second part First We recommend food If a old person is came on the basis of previous order It is done by face recoginition system and than in second part We record the food that we serving to cutsomer on the basis of food detection If a new person is come than we store person image in our database with mobile no and name and food that he order
My motivation is that it can be used to various places where fast food is served on the counter 
## File Descriptions <a name="files"></a>

There are three main foleders:
1. data
    - disaster_categories.csv: dataset including all the categories 
    - disaster_messages.csv: dataset including all the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
3. app
    - run.py: Flask file to run the web application
    - templates contains html file for the web applicatin

## Results<a name="results"></a>

1. A food detection model formation with 80% accuracy
2. A recommend food system from the image and on the basis of height, weight and Gender
3. A recommend food system for restaurant by face recoginition and live food detection model


## Licensing, Authors<a name="licensing"></a>

Credits must be given to the company who created Food 101 dataset and my mentor Saurabh Bhardwaj(https://www.linkedin.com/in/saurabh-bhardwaj-b97a1539/) who provide me to guidence in this project 
## Instructions:<a name="instructions"></a>
How to use this :
1. Download the model https://drive.google.com/file/d/1zAMm8Od_bgg678CMxqT41DHHTbjTD_4_/view?usp=sharing 
2 . Run the model

