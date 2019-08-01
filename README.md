# Food-AI-Recommender-System
### Table of Contents

1. [Summary and Motivation](#installation)
2. [File Descriptions](#files)
3. [Results](#results)
4. [Licensing, Authors](#licensing)
5. [Instructions](#instructions)

## Summary and Motivation <a name="installation"></a>
This is made by Food 101 dataset in which there are 101 categories of food are avaliable. This dataset contain 101000 image including train and test. You can download from here https://www.vision.ee.ethz.ch/datasets_extra/food-101/ .I make model using this dataset with 80% accuracy on validation data and 72% accuracy on test data  using Inception v3 neural network made by Google.
This project cosist of two parts in First part(cal_recommd_food) We detect the food than we recommend food on the basis of Height,Weight and Gender using calorires per weight of food.
In second part(restaur_recomnd_food) First We recommend food If a old person is came on the basis of previous order. It is done by face recoginition system and than in second part. We record the food that we serving to cutsomer on the basis of food detection. If a new person is come than we store person image in our database with mobile no and name and food that he order
. My motivation is that it can be used to various places where fast food is served on the counter and how it will be usefeul for food lovers and fitness freak persons.
## File Descriptions <a name="files"></a>

There are two main foleders:
1. cal_recommd_food
    - bmi_level.csv: data consist of obesity level  on the basis of height weight and gender
    - calorie_data.csv: data consist of calories of 101 food categories with weights in grams
    - recmd_food_image.py: file which take image to prediction
    - predict_food_live.py: live food detecion
    - haarcascade_frontalface_default.xml: open cv front face classifier
    - Some random images
2. restaur_recomnd_food
    - details_7.csv: file consist of customer information
    - fod_recmond_dbase.py:recommend food to customer
    - store_order_signup.py: store details of food in  database for old customer and also details of new customer
    - haarcascade_frontalface_alt2.xml:Opencv classifier to detect image

## Results<a name="results"></a>

1. A food detection model formation with 80% accuracy
2. A recommend food system from the image and on the basis of height, weight and Gender
3. A recommend food system for restaurant by face recoginition and live food detection model


## Licensing, Authors<a name="licensing"></a>

Credits must be given to the company who created Food 101 dataset and my mentor Saurabh Bhardwaj(https://www.linkedin.com/in/saurabh-bhardwaj-b97a1539/) who provide me to guidence in this project 

## Instructions:<a name="instructions"></a>
for using this you have a better web camera and You should be in good light and also ur face is distinguish from food u having and This works only on 101 Food categories
How to use this :
1. Download the model https://drive.google.com/file/d/1zAMm8Od_bgg678CMxqT41DHHTbjTD_4_/view?usp=sharing 
2. Download the repositry open up cal_recommd_food for first part change the path of model in both python file
3. Open up the restaur_recomnd_food run both fod_recmond_dbase , store_order_signup file in same order and change the path of model in these  files


