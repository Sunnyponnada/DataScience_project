# DataScience_project
This project is part of a larger effort at Waze to increase growth

# Project Title
**“Mchine Learning prediction of monthly Waze user churn.”**

# Background: 
**Waze’s free navigation app makes it easier for drivers around the world to get to where they want to go. Waze’s community of map editors, beta testers, translators, partners, and users helps make each drive better and safer.**

This project is part of a larger effort at Waze to increase growth. Typically, high retention rates indicate satisfied users who repeatedly use the Waze app over time. Developing a churn prediction model will help prevent churn, improve user retention, and grow Waze’s business. An accurate model can also help identify specific factors that contribute to churn and answer questions such as: 
•	Who are the users most likely to churn?
•	Why do users churn? 
•	When do users churn? 

# Project Overview
* Incorporate with our team members and solved the user churn rate by data processing and cleaning, Exploratory data analysis, statistical analysis.
* model developed with two algorithms
  1. Random Forest
  2. XGBoost
	model	precision	recall	F1	accuracy
0	RF cv	0.457163	0.126782	0.198445	0.818510
0	XGB cv	0.425932	0.170826	0.243736	0.811866
* While performing predicting validation data, notice that the scores went down from the training scores across all metrics, but only by very little. This means that the model did not overfit the training data.
* Just like with the random forest model, the XGBoost model's validation scores were lower, but only very slightly. It is still the clear champion.

### **Use champion model to predict on test data**

Now, use the champion model to predict on the test dataset. This is to give a final indication of how you should expect the model to perform on new future data, should you decide to use the model.
model	precision	recall	F1	accuracy
0	RF cv	0.457163	0.126782	0.198445	0.818510
0	XGB cv	0.425932	0.170826	0.243736	0.811866
0	RF val	0.445255	0.120316	0.189441	0.817483
0	XGB val	0.422680	0.161736	0.233951	0.812238
0	XGB test	0.423963	0.181460	0.254144	0.811189

### Confusion Matrix
![download](https://github.com/user-attachments/assets/617da96b-dbf3-4b5f-a3a2-91c0fe107d61)

# Business Understanding
Develop a machine learning model to predict user churn. Churn quantifies the number of users who have uninstalled the Waze app or stopped using the app. This project focuses on monthly user churn. An accurate model will help prevent churn, improve user retention, and grow Waze’s business.

# Data understanding
* Data containing 194987rows of users using Waze application.
* dtypes: float64(3), int64(8), object(2)

#### **`sessions`**
_The number of occurrence of a user opening the app during the month_
![download](https://github.com/user-attachments/assets/90ddc8fe-b058-4406-b057-52b8563d2f4a)
![download](https://github.com/user-attachments/assets/423b6474-05f8-475f-a35f-dd07bf1d7f04)
The `sessions` variable is a right-skewed distribution with half of the observations having 56 or fewer sessions. However, as indicated by the boxplot, some users have more than 700.

#### **`drives`**
_An occurrence of driving at least 1 km during the month_
![download](https://github.com/user-attachments/assets/26b295aa-0786-44b3-9ec6-bdcee6296455)
![download](https://github.com/user-attachments/assets/b601c995-a237-4b84-8073-6773e6cd1aa3)
The `drives` information follows a distribution similar to the `sessions` variable. It is right-skewed, approximately log-normal, with a median of 48. However, some drivers had over 400 drives in the last month.

#### **`total_sessions`**
_A model estimate of the total number of sessions since a user has onboarded_
![download](https://github.com/user-attachments/assets/cc8d5c9b-15a7-43de-a3eb-5d128a34d074)
![download](https://github.com/user-attachments/assets/b2e7bbfd-ecb2-4f5e-b6c2-e8823e597f50)
The `total_sessions` is a right-skewed distribution. The median total number of sessions is 159.6. This is interesting information because, if the median number of sessions in the last month was 48 and the median total sessions was ~160, then it seems that a large proportion of a user's total drives might have taken place in the last month. This is something you can examine more closely later.

#### **`activity_days`**
_Number of days the user opens the app during the month_
![download](https://github.com/user-attachments/assets/d89f1f16-dcd6-42c7-9edd-88403b842162)
![download](https://github.com/user-attachments/assets/4eb343a4-8bc9-4b3a-b802-727931c4aaba)
Within the last month, users opened the app a median of 16 times. The box plot reveals a centered distribution. The histogram shows a nearly uniform distribution of ~500 people opening the app on each count of days. However, there are ~250 people who didn't open the app at all and ~250 people who opened the app every day of the month.

This distribution is noteworthy because it does not mirror the `sessions` distribution, which you might think would be closely correlated with `activity_days`.

#### **`device`**
_The type of device a user starts a session with_
![download](https://github.com/user-attachments/assets/7e3254c4-de8b-4797-aa7d-20cce06a6efb)
There are nearly twice as many iPhone users as Android users represented in this data.

Analysis revealed that the overall churn rate is \~17%, and that this rate is consistent between iPhone users and Android users.

Perhaps you feel that the more deeply you explore the data, the more questions arise. This is not uncommon! In this case, it's worth asking the Waze data team why so many users used the app so much in just the last month.

Also, EDA has revealed that users who drive very long distances on their driving days are _more_ likely to churn, but users who drive more often are _less_ likely to churn. The reason for this discrepancy is an opportunity for further investigation, and it would be something else to ask the Waze data team about.

There is missing data in the user churn label, so we might need further data processing before further analysis. -There are many outlying observations for drives, so we might consider a variable transformation to stabilize the variation. -The number of drives and the number of sessions are both strongly correlated, so they might provide redundant information when we incorporate both in a model. -On average, retained users have fewer drives than churned users.

* Nearly all the variables were either very right-skewed or uniformly distributed. For the right-skewed distributions, this means that most users had values in the lower end of the range for that variable. For the uniform distributions, this means that users were generally equally likely to have values anywhere within the range for that variable.
* Most of the data was not problematic, and there was no indication that any single variable was completely wrong. However, several variables had highly improbable or perhaps even impossible outlying values, such as `driven_km_drives`. Some of the monthly variables also might be problematic, such as `activity_days` and `driving_days`, because one has a max value of 31 while the other has a max value of 30, indicating that data collection might not have occurred in the same month for both of these variables.
* Users of all tenures from brand new to \~10 years were relatively evenly represented in the data. This is borne out by the histogram for `n_days_after_onboarding`, which reveals a uniform distribution for this variable.

# Conclusion
> _Logistic regression models are easier to interpret. Because they assign coefficients to predictor variables, they reveal not only which features factored most heavily into their final predictions, but also the directionality of the weight. In other words, they tell you if each feature is positively or negatively correlated with the target in the model's final prediction._
> _Tree-based model ensembles are often better predictors. If the most important thing is the predictive power of the model, then tree-based modeling will usually win out against logistic regression (but not always!). They also require much less data cleaning and require fewer assumptions about the underlying distributions of their predictor variables, so they're easier to work with._
> _New features could be engineered to try to generate better predictive signal, as they often do if you have domain knowledge. In the case of this model, the engineered features made up over half of the top 10 most-predictive features used by the model. It could also be helpful to reconstruct the model with different combinations of predictor variables to reduce noise from unpredictive features._
> _It would be helpful to have drive-level information for each user (such as drive times, geographic locations, etc.). It would probably also be helpful to have more granular data to know how users interact with the app. For example, how often do they report or confirm road hazard alerts? Finally, it could be helpful to know the monthly count of unique starting and ending locations each driver inputs._





