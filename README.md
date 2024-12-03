# Exercise-7---Regression-

Exploratory Data Analysis On Google Playstore

Google Playstore, formerly Android Market, is a digital distribution service operated and developed by Google. It serves as the official app store for certified devices running on the Android operating system, allowing users to browse and download applications developed with the Android software development kit (SDK) and published through Google.

Android devices directly links up with the Google Playstore from where users are able to download the wide varieties of apps.

India is the highest consumer of Android Devices and all categories of people uses Android devices. Also app developments are also done for the Google Playstore the most as it provides the perfect platforms for the developers.


I got the dataset from https://www.kaggle.com/ which is a trusted open source dataset collection hub.
This datset contains the following features:
•	App: Application name

•	Category: Category the app belongs to

•	Rating: Overall user rating of the app by users

•	Reviews: Number of user reviews for the app after usage

•	Size: Size of the app

•	Installs: Total number of downloads of the particular application

•	Type: If the app in the play store is paid or free

•	Price: Price of the app

•	Content Rating: Age group the app is targeted at - Children / Mature 21+ / Adult


•	Genres: An app can belong to multiple genres (apart from its main category). For eg, a musical family game will belong to Music, Game, Family genres.

•	Last Updated: Date when the app was last updated on Play Store

•	Current Ver: Current version of the app available on Play Store

•	Android Ver: Min required Android version


Downloading the Dataset :
Let's begin by downloading the data, and listing the files within the dataset.
dataset_url = 'https://www.kaggle.com/lava18/google-play-store-apps'

### Linear Regression :


Aim : 	

The Aim is to build and evaluate five different linear regression models using the Google Play Store apps dataset. It demonstrates how feature selection impacts model performance in predicting app ratings. The code includes data preprocessing, training models, calculating evaluation metrics, and visualizing the relationship between actual and predicted ratings.




CODE :

![image](https://github.com/user-attachments/assets/b290cb24-db6c-47ec-96fa-72f40e7544ed)
![image](https://github.com/user-attachments/assets/6a3a1ed2-e13f-4e33-be23-262132fe25f9)
![image](https://github.com/user-attachments/assets/4032c883-44c2-41f0-801e-7e876e633fd5)
 
Output :
Model 1 – 

![image](https://github.com/user-attachments/assets/4013ee69-45bd-4873-a621-fc013eabd31f)
![image](https://github.com/user-attachments/assets/f8dd37da-0b73-4406-99c2-edb112539016)

 

Model 2 –

![image](https://github.com/user-attachments/assets/e0b41ea2-852b-4f58-913a-b932bc5cd83c)
![image](https://github.com/user-attachments/assets/9ca4a30d-5f93-4dcb-bcf2-ee521b75e851)



Model 3 –

![image](https://github.com/user-attachments/assets/3ab1ea93-6ea1-4058-832d-d1f30901a08a)
![image](https://github.com/user-attachments/assets/e6f3eb53-c613-49a3-b586-e6cb5347caf9)


 

Model 4 –

![image](https://github.com/user-attachments/assets/9c0fdfe9-3bdc-410e-bedc-33c2d62d35d8)
![image](https://github.com/user-attachments/assets/9321bb7a-2c95-4ce5-8ebf-427c137368f6)


 
Model 5 –

![image](https://github.com/user-attachments/assets/174cd358-aa3a-49a5-8da7-657806ec12a7)
![image](https://github.com/user-attachments/assets/27589f5b-2da2-4246-a673-3a0087111263)


 
### LOGISTIC REGRESSION :

AIM :

The aim of this code is to implement a Logistic Regression model for binary classification of apps in the Google Play Store dataset. The goal is to predict whether an app has a high rating (i.e., a rating greater than or equal to 4) or a low rating (i.e., a rating less than 4). The code preprocesses the data, selects relevant features, trains the model, evaluates its performance, and visualizes the results.

CODE :

![image](https://github.com/user-attachments/assets/8f873ede-9e17-4b02-ba39-8c5cbed1fbf3)
![image](https://github.com/user-attachments/assets/8f701fbf-910d-416f-8a0e-d41304fad1e0)
![image](https://github.com/user-attachments/assets/0415d79c-01b7-4f97-b926-9927209ba19b)
![image](https://github.com/user-attachments/assets/f995cfbf-e009-4923-9367-d53ebba978df)

 
    
OUTPUT :

![image](https://github.com/user-attachments/assets/082d7c4f-7953-4b65-88fe-4ea475f7a233)
![image](https://github.com/user-attachments/assets/d5782fff-fd8f-4917-ba91-91d679d2370e)


  

### KNN :

AIM :

The aim of this code is to build a K-Nearest Neighbors (KNN) classifier to predict whether an app in a dataset has a high rating or low rating. Specifically, the goal is to:
1.	Preprocess the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
2.	Train a KNN model to classify apps based on features like Category, Reviews, Size, Installs, and Price.
3.	Evaluate the model's performance using metrics such as accuracy and the confusion matrix, which will help us understand how well the model is distinguishing between high and low ratings.
4.	Improve the model by tuning hyperparameters like the number of neighbors (k) to find the best-performing model.
Specific Steps:
1.	Data Preprocessing:
•	Handle missing values by filling them with appropriate strategies like using the median for numerical columns.
•	Convert categorical columns into numerical formats using Label Encoding.
•	Scale numerical columns to normalize the data, as KNN is sensitive to the scale of the features.
2.	Model Training:
•	Split the dataset into training and testing sets.
•	Train the KNN classifier on the training data.
3.	Model Evaluation:
•	Test the model on unseen data (the test set) and calculate the accuracy.
•	Use a confusion matrix to understand how many predictions were correct or incorrect for each class (high or low rating).
4.	Output:   Accuracy Score: To see how well the model is performing.
•	Confusion Matrix: To see how many correct and incorrect predictions were made for each class. 
•	The overall objective is to build a robust classifier that can predict app ratings accurately, helping in analyzing app performance based on these features.

