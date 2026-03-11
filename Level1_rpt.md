# **Level 1 Report**
---
# Task 1: MATLAB ML Onramp Course

The task was to complete the MATLAB Machine Learning course, which teaches the end-to-end workflow of machine learning, from data importing to evaluation. This particular course focuses on a project classifying handwritten alphabets.

## Course Modules and Learnings:

###  Overview of Machine Learning

Understood what machine learning is and how it is applied.

In this course, I work around classification models, the models that put data into categories, like cat, dog, or penguin.By trying to classify handwritten letters, one will go through the full ML workflow: import data, data preprocessing,build a model, validation and evaluation.


### Import Data, Extract Features, and Partition Data
Learned how to organize and import handwritten letter data into MATLAB. Since machine learning models need structured inputs, we extracted meaningful characteristics from each letter, like height, width, and writing duration. To make this process efficient, functions were created to automatically extract features from all files at once, which helped handle large datasets consistently.

Next, the data into training and validation sets to make sure our models could generalize well to new letters. This prevents overfitting, where a model performs perfectly on training data but poorly on unseen data. and also explored cross-validation, which evaluates the model on multiple subsets to get a more robust measure of accuracy and reliability.

### Train Models, Evaluate Performance, and Improve Performance

Trained different supervised learning models, like decision trees and k-nearest neighbors (KNN), using MATLAB's Classification Learner. Observed how each model performed with the same data and learned that model choice can affect accuracy.

Evaluated model performance using accuracy metrics and confusion matrices, which helped us see which letters were being misclassified and understand patterns in errors. This gave insight into how the model could be improved.

Finally, explored ways to improve performance, such as tuning hyperparameters, refining feature extraction, or handling variations in handwriting styles. These techniques help make the model more accurate and robust when predicting letters it hasn't seen before.

### MATLAB vs Python (scikit-learn):

- MATLAB provides a more visual and app-based approach, reducing the need for manual coding.

- WHile Python offers greater flexibility and customization

- MATLAB is helpful in rapid prototyping.

- But again they both follow the same conceptual pipeline: data preparation --> training --> validation --> testing.

![Certificate](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Matlab1.png)

---
---

# Task 2: Kaggle Crafter – Build & Publish Your Own Dataset


The task was to create a dataset from scratch and to make sure the dataset is clean and organized and publish it on Kaggale. The purpose of the task is to understand the what makes a dataset usable and easy for others to work with, and as well as to understand the effort that goes into data collection, and problems associated with it and the  different methods used for data collection, documentation and publishing a dataset.

For this task, I created a synthetic heart disease dataset. The dataset is termed *synthetic* because the dataset is created using Faker library to generate realistic but fake data.

[data collection challenges](https://easy-feedback.com/blog/data-collection/problems-with-data-collection/)

## Learning Outcomes

- The first and most significant challenge in any kind of data collection would be the source. For example, in the case of medical datasets, this becomes even more difficult due to privacy concerns, limited accessibility, and the amount of effort required to collect and organize the data. Because of this, data collection in itself becomes the most challenging part of the ML workflow.

- So when in situations where we need realistic and sensitive data but cannot afford to collect it in real life, we can rely on synthetic data generation libraries such as Faker and DataDraw to create usable datasets for practice and experimentation.

- Another important aspect of dataset creation is the cleaning and organization of raw data. Mostly the collected data is unstructured and messy, and such data cannot be directly used for training machine learning models as worsens model performance. Again a significant amount of the time is spent on data preprocessing and cleaning. Library such as Pandas help in transforming raw data into a clean, structured, and usable format.

- Next, while publishing a dataset, proper documentation is of utmost importance. Clear descriptions, metadata, and column explanations are must as it makes easier for others to understand the dataset, explore its contents without needing to download it.

- An additional requirement of the task was to publish the dataset on Kaggle and achieve a usability score greater than 8.5. This specifically helped in understanding the factors that contribute to a high-scored dataset on Kaggle, such as a detailed description, appropriate tags, a suitable license, and well-structured metadata.

Here is the dataset I created using the Faker library.

[Dataset](https://www.kaggle.com/datasets/vismayag/synthetic-heart-disease-dataset)

And i got the Usability Score of *9.41*

- ---
---

- Used Faker to generate random but realistic values like names, gender, and patient IDs.
- Used Datagram / Pandas to structure all the data into a clean .csv file.
- Created 500 synthetic patient records with features like chest pain type, blood pressure, cholesterol, fasting blood sugar, ECG results, thalassemia, exercise-induced angina, and heart disease presence.
- Added metadata when uploading to Kaggle: description, tags, license (CC0), and a cover image.

---

### About Faker:

Faker is a Python library that lets you generate fake but realistic data. It can create names, addresses, phone numbers, emails, dates, and lots of other information. Faker is really useful when you want to simulate a real dataset safely, for example for testing or practicing data science skills, without using any sensitive or personal data. Basically, it helps you make your dataset look realistic even though it’s completely synthetic.

---
---
---

# Task 3: Data Cleaning and Pre-processing using Pandas

The task was to clean a raw and messy dataset and convert it into a usable form using the Pandas library in Python.

Real-world datasets are often messy, unstructured and contain a large number of missing values, duplicate entries, and invalid entries. If such data is directly fed into machine learning models, it can lead to garbage outputs and lead to poor model performance. Hence data cleaning and pre-processing becomes the fundamental step in any machine learning workflow.

Pandas is one of the most widely used library for data cleaning because it provides built-in methods to handle common data quality issues such as missing values, duplicate records, and inconsistent data formats etc. and also because it is capable of handling large datasets efficiently, which can be a limitation when using tools such as Excel.

In addition, Pandas integrates well with other Python libraries such as NumPy for numerical operations, Matplotlib and Seaborn for visualization, and scikit-learn for machine learning, which enables the smooth end-to-end data workflow. and it also supports reading and writing data from multiple formats including CSV, Excel, JSON, and SQL databases, making data import and export easier.

---

##  About the Dataset Used

The given dataset is of a customer-related data and contains 51,000 records. It includes 10 columns representing various customer attributes such as CustomerID, Name, Age, Gender, Country, Preferred Device, Signup Date, Login Date, Email, and Total Purchase.

---

##  Data Cleaning and Pre-processing

With the initial data check using Pandas functions such as `head()` and `info()`, it was observed that some columns had incorrect data types. For example, the Signup Date and Login Date columns were stored as object (string) types instead of proper datetime format and therefore required conversion. The check also revealed the presence of missing values across multiple columns, indicating that the dataset needed thorough cleaning and preprocessing before further analysis.

 ## Exploratory Data Analysis (EDA)

 Through EDA, most of the issues in the dataset were identified. It showed the columns with highest number of missing values and also revealed the presence of invalid age values that were outside realistic ranges(e.g some are 200 years old while, some are negatively aged).

 It also showed the mistakes in categorical columns such as Country, Gender, and Preferred Device. In the Total Purchase column, some entries were negative values, which are not logically valid and required correction.

 These are all the issues that in this dataset that has to be fixed through data cleaning using the Pandas library

---

Data cleaning and pre-processing was performed based on the issues found during EDA. 

- First, duplicate entries were removed from the dataset using the `drop_duplicates()` function to ensure that each entry was unique.

- Next, the Gender, Country, and Preferred Device columns were cleaned. All three columns had similar issue that was inconsistent formatting and spelling errors. These were corrected using the `replace()` function to replace wrongly spelled entries with consistent format.

- Then missing values in these columns were identified using the `isna()` function and were filled with  `"Unknown"` using the `fillna()` function.

- As for the Age column, few unrealistic age values were were observed during EDA. So a realistic age range was defined, and any age values outside this range were considered invalid and they are set to null.As for the missing age values, they were filled using the median age calculated per Gender and Country using the `median()` function. 

- As for the Signup Date and Login Date columns, I observed inconsistencies in date formats, where some entries were in M/D/Y format while others were in D/M/Y format. Hence to standardize these columns, the `to_datetime()` function was used to convert them into proper datetime format.

- Additionally, the Login Date column was checked for the presence of any future login dates using `timestamp.today()`, as future logins are not logically possible. 

- Next, for the Total Purchase column, EDA revealed the presence of negative values, which are not logically valid.So these negative values were converted to positive values using the `abs()` function. And any missing values in this column were then filled using the median.

- As for the Name column, some entries contained numbers and special characters. These were removed while keeping only the valid characters such as alphabets, hyphens, and apostrophes. The `strip()` function was used to remove extra spaces before and after the names. The names were then standardized to a first-name and last-name format, and titles were removed as they were not consistent across the dataset.

- For the CustomerID column, the `strip()` function was used to remove extra spaces and ensure consistent formatting. Missing CustomerID values were replaced with null values.
( Both Name and CustomerID columns are unique  ID columns, hence we can't replace missing values with a random placeholder.)

- For the Email column, all entries were first converted to lowercase using the `str.lower()` function. Emails with missing usernames (starting with `@`) were handled by placing a placeholder before the `@` symbol. Missing email values were replaced with null values, and emails with incorrect formats were identified and handled accordingly.

---

## Learning Outcomes

This task showed the importance of having a clean dataset and the amount of time and effort that goes into data cleaning.

It also showed how EDA helps in identifying most of the problems with minimal effort, as patterns in the data become clearly visible.

Another key learning was that handling missing values is not a one-size-fits-all process; it depends on the dataset and the model we are working with, and null values cannot be filled blindly.

It also became clear that even within the same dataset, the same cleaning strategy cannot be applied to all columns.




## Final Dataset Summary

- Duplicate records were removed, improving data reliability.
- Missing values were identified and handled using appropriate imputation techniques.
- Invalid age values were corrected using group-wise median imputation.
- Date columns were standardized and logically invalid future login dates were identified.
- Negative purchase values were corrected, and remaining missing values were filled using the median.
- Text-based columns were cleaned and standardized using string operations and pattern matching.

The final dataset is consistent, reliable, and ready for further analysis or machine learning tasks.

[Cleaned_dataset](https://drive.google.com/open?id=1H_3s3W5F4kor1JNBPhM-2R62ATn4hz6k&usp=drive_fs)

# Task 4 : Anomaly Detection

The objective of this task is to learn and apply anomaly detection techniques to identify unusual patterns and outliers in the given dataset. This task involves using at least two approaches, one from statistical method and one from unsupervised anomaly detection method.

For each method, the anomalies had to be identified and then compared. Based on the comparison, the top five anomalies in the dataset were determined.

---
## Learning Outcomes

- This task helped in understanding that anomaly detection in machine learning is used to identify data points that do not follow the normal patterns present in a dataset.

- And this very concept help identify critical issues such as fraud, system failures, security breaches, or incorrect outputs, which makes anomaly detection an important step in real-world applications.

- Anomaly detection models first learn the normal behaviour from the data and then identify observations that significantly deviate from it.

---

For this task, a dataset related to G-Flix Inc. was given, containing information about employees, their login activity, access status, and work-related behaviour. G-Flix Inc. suspected an internal security breach, but since the exact nature of the anomaly was unknown, anomaly detection techniques were required to analyze user activity patterns and identify potential internal suspects involved in the breach.

## Dataset Description

 The given dataset consists of 505 entries with six columns. The columns include timestamp, User ID, login duration (in minutes), data accessed (in MB), number of files downloaded, and remote access status.

 The dataset does not contain any predefined anomaly labels.So it is not known beforehand which data points represent normal behavior and which represent anomalous behavior. 

## Initial Data Exploration

During the initial exploration phase, the numeric features were analyzed to understand their ranges and extreme values. The maximum values for Login Duration, Data Accessed, and Files Downloaded were checked to get an overall view of the data distribution. This helped in building an intuition about what potential anomalies might look like before applying any detection techniques.

## Understanding Normal Behavior

Before applying anomaly detection, it was important to understand what normal behavior looks like in the dataset as anomalies are the deviation from this behaviour.

To understand this, visualizations were used. Histograms for Login Duration, Data Accessed, and Files Downloaded gave a an idea of how most of the values were distributed and also pointed out the entries that were very different from the rest.

But looking at each feature separately was not enough to fully understand the anomalies. So feature-wise plots were used to study the relationship between Login Duration, Data Accessed, and Files Downloaded. From this, it became clear that most users had moderate login durations (around 5–65 minutes), accessed less than 400–500 MB of data, and downloaded fewer than 10 files. Only a small number of users showed extreme behaviour such as very long login durations, unusually high data access, and excessive file downloads. These were then taken forward for anomaly detection.

## Anomaly Detection Methods

After understanding the normal behavior patterns in the dataset, next part of the task is to apply anamoly detection methods.

 In this task, two approaches to anomaly detection were used:
- statistical methods and
- unsupervised machine learning methods.

For statistical anomaly detection, the Z-score method and the Interquartile Range (IQR) method were applied. For unsupervised anamoly detection, the Isolation Forest algorithm was used.

---

### Z-Score Method

The Z-score method is a statistical technique used to identify values that significantly deviate from the rest of the data. It works by calculating how far a data point is from the mean of the dataset in terms of standard deviation.

The Z-score is calculated using the formula:

$$
Z = (X − μ) / σ 
$$ 

where,

- X represents an individual data value
- μ is the mean of the feature
- σ is the standard deviation of the feature.

This method is based on the principle that most normal data points lie close to the mean, while extreme values that are far from the mean are considered to be the anomalies.

If a data point has a Z-score that is very high or very low, it indicates that the value is unusual compared to the rest of the dataset.

In practice, we chose a threshold value to decide when a data point should be considered an anomaly. Commonly used thresholds are ±2 or ±3. For this task,I've used my threshold as ±3. Using a threshold of ±2 resulted in too many data points being flagged as anomalies, making the model overly sensitive. Hence I changed the threshold to ±3, so that only extremely ususual get flagged as anomolies resultinng in less sensitive and more reliable anamoly detection model.

---

### Interquartile Range (IQR) Method

The second statistical method that i've used is the Interquartile Range (IQR) method. Unlike the Z-score method, IQR does not rely on the mean and standard deviation. Instead, it looks at how the middle 50% of the data is spread, so extreme values do not affect it much.

To calculate the IQR, the first quartile (Q1) and third quartile (Q3) are to be calculated. Q1 represents the 25th percentile of the data, meaning 25% of the data lies below this value. Q3 represents the 75th percentile, meaning 75% of the data lies below this value. The difference between Q3 and Q1 gives the interquartile range (IQR).

![IQR](https://ars.els-cdn.com/content/image/3-s2.0-B9780123814791000022-f02-02-9780123814791.jpg)

$$ IQR = Q3 -Q1 $$

Once the IQR is calculated, lower and upper bounds are defined to identify anomalies. Any data point that falls below the lower bound or above the upper bound is considered an anomaly, while values within this range are treated as normal.

$$      Lower Bound = Q1 − Th × IQR $$

$$ Upper Bound = Q3 + Th × IQR  $$

The IQR method is useful because it captures the normal middle range of the data without getting affected by extreme outliers, which makes it more robust than the Z-score method in certain cases. Similar to the Z-score approach, its sensitivity can also be adjusted by changing the threshold value, a higher threshold makes the detection stricter, while a lower threshold makes it more sensitive to anomalies.

---

### Unsupervised Anomaly Detection – Isolation Forest

The previous methods that discussed were statistical anomaly detection techniques. The next approach used in this task is an unsupervised anomaly detection method.

In unsupervised learning, the dataset doesn't contain labels indicating normal or abnormal behavior, and the model learns patterns directly from the data to identify the labels.

For this task,I've used Isolation Forest as the unsupervised anomaly detection technique. and most importantly before applying unsupervised anomaly detection technique, numeric features must be scaled since they exist in different ranges, such as Login Duration and Files Downloaded exists around in hundreads,while Data Accessed in thousands. Feature scaling ensures that no single feature dominates the model due to its magnitude.

Isolation Forest detects anomalies by isolating data points instead of first trying to model normal behaviour. The idea behind this is that anomalous points are easier to separate from the rest of the data, so they require fewer splits in randomly generated trees, while normal points take longer paths because they are similar to many other values. Hence, shorter path lengths indicate anomalies.

## Comparison of Anomalies and Identification of Top Suspects

After applying the anomaly detection methods, the last step of the task was to compare the data points that were flagged as anomalies across different techniques and identify the most suspicious users. Since each anomaly detection method works differently, they do not always flag the exact same data points as anomalies. However, when a data point is flagged by more than one method, it becomes more suspicious.

To combine and compare the results from all the methods, an anomaly score was assigned to every data point. For each anomaly detection method (Z-score, IQR, and Isolation Forest), a data point received a score of 1 if it was flagged as an anomaly and 0 if it was not. These scores were then summed across all the methods to obtain a total anomaly score for each data point. Data points with higher total anomaly scores were considered as the **top suspects**.

By sorting the data points based on the total anomaly score in descending order, the top suspects are:

- **User_033**  
  - This user was flagged by all three methods. 
  - The login duration was approximately 200 minutes, while the data accessed was only 5 MB and the number of files downloaded was 100.
  - Very high file downloads with extremely low data access and a very long login duration is highly abnormal and does not match normal usage patterns.

- **User_045**  
  - This user was also flagged by all three methods. 
  - The login duration was only 3 minutes, while the data accessed was around 4,500 MB, with no files downloaded.
  - Extremely high amount of data accessed in such a short session duration is unusual and strongly deviates from normal behavior.

- **User_036**  
  - This user has a login duration of approximately 300 minutes, data access of around 5,000 MB, and about 50 files downloaded. 
  - This combination of very long session duration, extremely high data access, and high file downloads is abnormal. 
  - Again, this user was flagged by all three anomaly detection methods.

- **User_032**  
  - This user had a login duration of around 120 minutes, data access of approximately 4,000 MB, and about 60 files downloaded. 
  - The unusually high data access and file downloads over a long session duration makes this user to be flagged by all three methods.

- **User_025**  
  - This user was flagged by the IQR and Isolation Forest methods.
  - The login duration was around 5 minutes, data access was approximately 30 MB, and no files were downloaded. 
  - While not as extreme as the other cases, this behavior deviates from the most of normal users.

The complete implementation for this task is available in the notebook below:

 [Anomaly Detection – Kaggle Notebook](https://www.kaggle.com/code/vismayag/anomaly-detection-task)

 ---

# Task 5: Logistic Regression from Scratch

The objective of this task was to understand binary classification and the working of the logistic regression model by implementing it from scratch. In addition to the manual implementation, a logistic regression model from the scikit-learn library was also trained, and both models were compared and evaluated based on their performance.

## Dataset Description and Pre-processing

The dataset used for this task was a heart disease prediction dataset. The features included gender, age, education, current smoking status, number of cigarettes per day, use of blood pressure medication, history of prevalent stroke, presence of hypertension, diabetes status, total cholesterol, systolic blood pressure, diastolic blood pressure, body mass index (BMI), heart rate, and glucose level.

The target variable was the 10-year CHD indicator, which represents whether a person is likely to develop coronary heart disease within the next 10 years.

Several columns contained missing values, which were handled before model training. For binary categorical features, missing values were filled using the mode. For continuous numerical features such as cholesterol, blood pressure, BMI, and glucose, missing values were filled using the median.

## Mathematics Behind Logistic Regression

Logistic regression is used for binary classification problems where the output can take only two values, typically 0 or 1. Although it is used for classification, it is called a *regression* model because it first computes a continuous value, which is then converted into a probability between 0 and 1 using the logistic (sigmoid) function. This probability is later converted into a binary output using a decision threshold.

As discussed earlier, the model first computes a linear combination of the input features:

$$
z = w. x + b
$$

where:
- w  is the weight vector  
- x represents the input features  
- b is the bias term  

Since the value of z can range from negative to positive infinity, it cannot be directly interpreted as a probability.

### Sigmoid Function

To convert z into a probability, the sigmoid function is applied:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid function maps any real-valued input to a range between 0 and 1.

- Large positive values of z give outputs close to 1  
- Large negative values give outputs close to 0  
- When z=0, the output is 0.5  

### Loss Function

To measure how well the model performs, the Binary Cross-Entropy Loss (Log Loss) is used. This loss function measures the difference between the actual class labels and the predicted probabilities.

For a single data point:

$$
L = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]
$$

where:
- y is the actual label  
- $ \hat{y} $ is the predicted probability  

The total loss is given by averaging the loss across all data points.

*The logarithmic nature of this loss penalizes confident but incorrect predictions more heavily than uncertain ones.*

### Gradient Descent

To minimize the loss, the model parameters (weights and bias) are updated using gradient descent. The gradients of the loss with respect to the weights and bias are computed, and the parameters are updated iteratively as:

$$
w = w - \alpha \frac{\partial L}{\partial w}
$$

$$
b = b - \alpha \frac{\partial L}{\partial b}
$$

where $ \alpha  $ is the learning rate.

This process is repeated until the loss decreases and the model converges.

After training, predicted probabilities are converted into class labels using a decision threshold. If the predicted probability is greater than or equal to the threshold, the output is classified as 1, otherwise, it is classified as 0.

## Logistic Regression Implementation from Scratch

For the from-scratch implementation, all the core mathematical components of logistic regression were explicitly defined, including the sigmoid function, loss function, gradient computation, and parameter updates using gradient descent. A prediction function was also implemented to convert predicted probabilities into binary class labels based on the chosen threshold.

## Model Evaluation of Scratch Implementation

The scratch implementation was first evaluated using the default decision threshold of 0.5 and the results were:

Accuracy: 0.671  
Precision: 0.250  
Recall: 0.581  
F1-score: 0.350  

At this threshold, the model provides a balanced performance. But, in medical diagnosis, recall is the most important metric because missing a true CHD case is more critical than stating a false positives.

### Best Threshold Selection using ROC Curve

To get a more suitable threshold value, the ROC curve was plotted and the optimal threshold was calculated using Youden’s Index (TPR − FPR).

The obtained optimal threshold is 0.347.

When the performance at this threshold is evaluated, the results were:

Accuracy: 0.5094  
Precision: 0.22243  
Recall: 0.891  
F1-score: 0.356  

Therefore lowering the threshold significantly increased recall from 0.581 to 0.891, that means the model can detect majority of CHD cases. but this came at the cost of reduced accuracy and precision, as more samples were classified as positive.

For medical datasets/models, this trade-off is acceptable, because identifying high-risk patients is more important than maintaining high overall accuracy.

---

## Scikit-learn Logistic Regression Evaluation

Accuracy: 0.670  
Precision: 0.249  
Recall: 0.581  
F1-score: 0.349  

The performance of the Scikit-learn model is almost identical to the scratch implementation at the default threshold. This suggests that the scratch implementation was mathematically correct.

---

## Observations

- The scratch model at threshold = 0.5 matches the Scikit-learn implementation, that is it shows the correctness of the implementation.
- Lower threshold values improves recall of the model.
- The F1-score improves slightly, indicating that there's a better balance between precision and recall.

---

## Handling Imbalanced Data

In imbalanced datasets, models show bias towards predicting the majority calss. This results in higher accuracy score but low recall.

In Scikit-learn, this problem can be handled by using,

*class_weight = 'balanced'*

While in the scratch implementation, class weights were applied by multiplying the error term during gradient calculation.

This increases the penalty for misclassifying minority-class samples, shifts the decision boundary toward the positive(minority) class, and hence improving recall.

---

## ROC Curve and Optimal Threshold

Logistic regression  turns output probabilities into class labels using threshold value.

And A ROC curve can evaluate performance of the model across all thresholds by plotting *TPR vs FPR*

![ROC-Curve](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/LR2.png)

We can use Youden’s Index to select the optimal threshold, that is given by

J = TPR − FPR

---

## Precision–Recall Curve for Imbalanced Data

For imbalanced datasets, the Precision–Recall curve gives more information than the ROC curve because it focuses on the performance of the positive class.


![PR Curve](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/LR3.png)

It shows,

- how many CHD cases are correctly detected (recall)
- how reliable the positive predictions are (precision)

And a score named Averege Precision sums up this into a single number, whivh tells the model's ability detect positive case.

---

## Key learnings

This task provided a clear understanding of:

- the internal working of logistic regression
- gradient descent optimization
- the effect of decision threshold on model performance
- handling imbalanced datasets using class weighting
- selecting an optimal threshold using ROC analysis

 The complete implementation for this task are available in the notebook below:

 [Logistic Regression – Kaggle Notebook](https://www.kaggle.com/code/vismayag/logisticregressionfromscratch)

---

# Task 6 : Battle-Test Your Model - Support Vector Machines

The objective of this task was to learn and understand Support Vector Machines (SVMs), implement them using the scikit-learn library on the given dataset, and evaluate their performance. The second part of the task involvs introducing noise into the dataset, retraining the model, and observing how the model’s performance changed to analyze the robustness of the model under non-ideal conditions.

## About SVM

Support Vector Machines are supervised machine learning models used for both classification and regression tasks. In classification problems, SVM works by finding an optimal hyperplane that separates different classes with the maximum possible margin. This margin represents the distance between the hyperplane and the nearest data points from each class, known as the support vectors.

![SVM1](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-24%20223217.png)

![SVM2](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-24%20223233.png)

Compared to models such as logistic regression, SVMs often perform better in high-dimensional feature spaces and are more resistant to overfitting, especially when an appropriate kernel and regularization parameter are used.

## Dataset Description

The dataset used for this task was the Wine Quality dataset. It consists of 1,597 entries with 11 input features and one target column. Unlike the previous task, this is not a binary classification problem, as the target variable contains multiple discrete values representing wine quality scores.

The input features include fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. The target feature is *quality*, which indicates the quality rating of the wine.

The dataset did not contain any missing values. However, it had duplicate entries and they were removed during preprocessing. Because removing duplicates reduces redundancy in the data and is beneficial for models like Support Vector Machines, as they perform better on smaller datasets.

## Understanding Support Vector Machines

In SVM, each data point is represented as a point in an n-dimensional feature space, where n is the number of input features. Classification is performed by finding an optimal hyperplane that separates the data points belonging to different classes.

The main goal of an SVM is to create the best possible decision boundary that divides this space into different classes while maximizing the margin, which is the distance between the hyperplane and the closest data points from each class. These closest points are called support vectors, and they play a crucial role in defining the decision boundary.

There are two types of margins used in SVMs: hard margin and soft margin. In a hard margin SVM, no misclassification is allowed, which works only when the data is perfectly separable. In contrast, a soft margin SVM allows some margin violations, making it more suitable for real-world datasets that contain noise or overlapping classes.

Mathematically, the decision boundary of an SVM is represented as:

$$
w \cdot x + b = 0
$$

The margin size is given by:

$$
\frac{2}{||w||}
$$

This shows that maximizing the margin is equivalent to minimizing the magnitude of the weight vector $||w||$.

![SVM3](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-24%20223447.png)

![SVM4](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-24%20223502.png)

An important property of SVMs is that the final decision boundary depends only on the support vectors and not on the remaining data points. Because of this, SVMs are generally robust and less sensitive to outliers.

For datasets that are not linearly separable, SVMs use kernel functions to learn non-linear decision boundaries. Kernels implicitly project the data into a higher-dimensional feature space where a linear hyperplane can be found, allowing SVMs to model complex non-linear patterns effectively.


## Model Implementation and Baseline Evaluation

The Support Vector Machine (SVM) classifier was implemented using the scikit-learn library with a Radial Basis Function (RBF) kernel and a regularization parameter \( C = 1 \). The RBF kernel allows the model to capture non-linear relationships by internally mapping the input features into a higher-dimensional space.

The regularization parameter \( C \) controls the trade-off between maximizing the margin and minimizing classification errors. A value of \( C = 1 \) provides moderate regularization, allowing a small number of misclassifications while maintaining a sufficiently wide margin. This improves generalization and reduces the risk of overfitting.

All input features were standardized using z-score normalization before training. This step is essential for SVMs because the algorithm is distance-based and sensitive to differences in feature scale.

For the noise-free dataset, the performance of the model is:

Accuracy: 0.5919  
Recall: 0.5919  
F1-score: 0.5644

![SVM5](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/SVM1.png)
---

## Noise Robustness Analysis

To evaluate robustness against noise, gaussian noise is added at setps to the given dataset.

At each noise level the model was retrained using 5-fold cross-validation then performance metrics were averaged across these folds.

CV provides a more reliable estimate of performance than a single train–test split and allows to analyze the stability of the model under noisy conditions.

---

## Breakdown Point Identification

The baseline cross-validated accuracy of the model was 0.6019.  
The breakdown threshold was defined as 90% of the baseline performance that is 0.5417. The model’s accuracy dropped below this threshold at a noise level of 0.5. And the breakdown occurs at the noise level 1.0.

Since the data were standardized, this indicates that the model begins to fail when the noise standard deviation reaches approximately 50% of the original feature variability.

![SVM6](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/SVM2.png)

![SVM7](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/SVM3.png)

---

## Interpretation of Results

At low noise levels (0–0.2), the changes in both accuracy and F1-score are relatively small, indicating that the SVM model is robust to small levels of noises in the input data.

Beyond a noise level of 0.5, a sharp decline in performance is observed, showing that the model will underperform at large levels of noises. the breakdown point provides a quantitative measure of the model’s tolerance to data corruption  

[SVM Notebook](https://www.kaggle.com/code/vismayag/svmsnb)

---

## Conclusion

The robustness analysis shows that the SVM model maintains stable performance under low and moderate noise conditions but experiences a significant drop when the noise level exceeds 0.5. Therefore marginbased classifier can tolerate a certain amount of disturbances before failing.

---
---


# Task 7: Fairness Meets Functionality

The objective of this task was to build a decision tree classifier from scratch using the ID3 algorithm and evaluate its performance. The latter part of task also involved identifying the most influential features in the decision-making process and analyzing whether the model exhibits bias toward demographic attributes such as age and gender.

## Dataset Description

The dataset used for this task was the Utrecht Fairness Recruitment dataset, which contains 225 entries and 21 columns. The target feature is *Should-hire*, which indicates whether a candidate should be hired.

For training the model, only the features that are directly relevant for prediction were selected. The features used were **Strength**, **Speedtest**, **Lifttest**, **livesnear**, and **testresult**.

## Building the Decision Tree Using the ID3 Algorithm

The ID3 algorithm builds the decision tree by selecting the feature that provides the highest information gain at each step.

First, entropy is computed for the entire dataset to measure the level of impurity:

$$
H(S) = -\sum_{i} p_i \log_2(p_i)
$$

where \( p_i \) represents the probability of each class.

Next, the information gain for each feature is calculated. Information gain is the reduction in entropy obtained after splitting the dataset based on a feature. The feature with the highest information gain is selected as the root node because it produces the most effective split.

This process is applied recursively to the resulting subsets, allowing the tree to grow until a stopping condition is met. The tree stops growing when:

- all samples in a node belong to the same class, or  
- no further features are available for splitting.

The final leaf nodes represent the model’s predictions for new queries.

![ID3](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/ID3_4.jpeg)

## Model Evaluation

The decision tree model built using the ID3 algorithm had following performance:

- Accuracy - 0.8235  
- Precision- 0.7619  
- Recall - 0.6956  
- F1-score- 0.7272  

The results indicate a reasonably balanced model. But the performance improved only after introducing constraints to the algorithm such as maximum tree depth and limit on the number of splits. This is because controlling the tree growth prevents the model from becoming overly complex and hence can it is to memorize the training data.

---

## Fairness Analysis for Demographic Attributes

The second part of the task is conduct a fairness analysis for demographic parities and equal oppurtunities.

Demographic parity - it measures the proportion of positive predictions for each demographic group(Gender and Age).

![Demographic Parity](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/ID3_1.png)

The results were, for gender, the hiring rate for males was higher than for females, indicating the existence of bias in predictions even when the gender was not used as a training feature.

Similarly across the age groups, the “<25” category had a higher hiring rate compared to the “25–35” and “35+” groups. which suggests the model favors younger candidates.

---

### Equal Opportunity Analysis

![Equal Opportunity](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/ID3_2.png)

Equal opportunity evaluates whether qualified candidates from different groups are identified at the same rate (recall).

Equal Oppurtunity - it evaluates wheather a qualified candidate get hired irrespective of Gender and Age group.

From the analysis, for gender, female candidates had a recall of 1.0, while male candidates had a recall of approximately 0.61, indicating that qualified male candidates were more likely to be missed than a female candidate.

As for age groups, the “35+” category had the lowest recall, indicating that qualified older candidates were less likely to be correctly predicted as “Should-hire”.

---

### Feature Importance Analysis

Feature importance can be caluculated by counting how frequently each feature appeared as a splitting node in the decision tree.

- *Strength* and *Speedtest* appeared most often, making them the most important and influential features.  
- *testresult* and *livesnear* contributed less to the final decisions.

![Feature Importance](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/ID3_3.png)

The complete implementation for this task is available in the notebook below:

[ID3 Decision Tree & Fairness Analysis – Kaggle Notebook](https://www.kaggle.com/code/vismayag/id3desiciontrees)

 ---


# Task 8: KNN with Feature Ablation Study

The objective of this task was to learn amd implement the kNN algorithm using the Breast Cancer Wisconsin dataset and also to understand the importance of features through a feature ablation study.

---

## Dataset Description and Pre-processing

The dataset used for this task was the Breast Cancer Wisconsin dataset. It had 32 features. During preprocessing, the `id` column was dropped as it has no relevance for training the model and the target variable **diagnosis** was encoded as 1 for M(Malignant) and 0 for B(Benignin) hence replacing the datatype from object to int.

The dataset has features that are derived from the measurements of breast cell nuclei, and each represented by three statistical values that are *mean, Standard error, and worst*

Since KNN is a distance-based algorithm, feature scaling must be performed before implementing so that all features contribute equally to the distance computation.

---

## Working of KNN and Mathematical Background

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression. It is different from any other algorithm that was implemented so far as it does not build an explicit model during training, instead, it stores the training data and makes predictions based on similarity/distance between data points.

For a given test data value/point, the algorithm:

1. First computes the distance between the test value and all training values.
2. Selects the **K nearest neighbors** based on the chosen distance metric.
3. Lastly it assigns the class label to the test value by majority voting among the K neighbors.

### Distance Metric

Talking about the distance metric mentioned above the most commonly used distance metric is the **Euclidean distance**, and is given by:

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

where:

- $(x) $= test data point  
- $(y)$ = training data point  
- $(n)$ = number of features  

Since the distance depends on the magnitude of feature values, feature scaling is necessary in KNN to avoid features with larger ranges from dominating the distance calculation.

### Decision Rule

For classification, the predicted class $\hat{y}$ is:

$$
\hat{y} = \text{mode}(y_1, y_2, \dots, y_K)
$$

where $y_1, y_2, \dots, y_K$ are the labels of the K nearest neighbors.

---

## Model Training and Evaluation

The KNN classifier from the scikit-learn library was used implement the model with *k = 5* as a parameter. The dataset was divided into training and testing sets.

The baseline model was trained using all input features and evaluated using performance metrics,:

- Accuracy - 0.9561
- Precision - 0.9743
- Recall - 0.9047
- F1-score - 0.9382

The baseline model has high accuracy and precision, with slightly lower recall, indicating that while malignant predictions were highly reliable, a small number of malignant cases were missed.

---

## Feature Abalation

To analyze the importance and relavance feature on the model, we perform abalation study, for the task, I've done abalation study by dropping single features, and in groups based on their statistical types.


## Single-Feature Ablation Study

In the first stage of ablation study, one feature was removed at a time and the model was retrained using the same training and testing configuration. The performance metrics were tabulated and compared with baseline model.

It was observed that the removal of individual features did not significantly affect the performance in most cases.

![Results](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-22%20223602.png)

For the comparision I've only tabulated recall and f1 drop because they are the imporatnt metrics for medical datasets.

---

## Grouped Ablation by Statistical Type

Similarly, grouped ablation was performed by removing features based on their statistical type, that is,

- All **mean** features removed together  
- All **standard error (SE)** features removed together  
- All **worst** features removed together  

From the results:

- Removing the mean features caused the largest performance degradation, with accuracy dropping from 0.956 -> 0.921, recall from 0.905 -> 0.810, and F1-score from 0.938 -> 0.883. This indicates that mean features are highly important for the KNN classifier.

- Removing the SE features resulted in a slight improvement in performance (accuracy 0.965, recall 0.929, F1-score 0.951), showing that SE features contribute the least and may even introduce minor noise.

- Removing the worst features had almost no effect on accuracy and only a small drop in recall (0.905 -> 0.881) and F1-score (0.938 -> 0.937), indicating that worst-case features are useful but not as critical as the mean features.

![table](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-22%20224110.png)

---

## Grouped Ablation by Measurement

Next analysis by removing all three statistical representations (mean, SE, and worst) of each measurement at once. This allowed evaluation of the importance of each physical property as it is. 

Again, removing these features had no significant impact on the model performance.

![table2](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-24%20224721.png)

---

## Feature Importance

From the ablation study, it was clear that all features do not contribute equally to the classification. Some features could be removed without causing any noticeable drop in performance, which means they are either redundant or less informative. Using this observation, a reduced feature set can be formed that gives almost the same accuracy while making the model simpler and easier to interpret.

[kNN Notebook](https://www.kaggle.com/code/vismayag/knn-notebook)

## Conclusion

The KNN classifier showed strong performance on the Breast Cancer Wisconsin dataset when trained on normalized data. Removing individual features did not affect the performance much because many of them contain overlapping information. However, grouped ablation made it clear that worst-case measurements and shape-related features play the most important role in detecting malignant tumors.

# Task 9: Evaluation Metrics – Pick the Best Performer

The objective of this task was to learn how to use joblib to load pretrained machine learning models and to evaluate and compare multiple models using a common test dataset in order to identify the best-performing model based on standard evaluation metrics.

---

## Dataset Description and Pre-processing

The Iris dataset was used as test dataset for evaluation. It consists of four numerical input features and a target variable with three classes.

The dataset was loaded using pandas, and the same dataset was used as input for all the pretrained models to ensure a fair and consistent comparison.

---

## Loading the Pretrained Models

The pretrained models were stored as *".pkl"* files and were loaded using the joblib. This allows faster evaluation of model performance on a new dataset without retraining.

Using these pretrained models, the Iris dataset was used for prediction and evaluation for five different modelsthat are Logistic Regression, Support Vector Machine (SVM), Decision Tree, kNN, and Random Forest.

---

## Importance of Evaluation Metrics

Evaluation metrics are essential for understanding the true performance of a machine learning model. The most commonly used metrics are Accuracy, Precision, Recall, and F1-score. Individually, these metrics do not fully describe model performance because they are interdependent and capture different types of errors.

Precision and recall provide deeper insight into the model’s behavior, while the F1-score gives a balanced measure of performance. Using multiple metrics ensures a fair and reliable comparison between models.

Hence, selecting appropriate evaluation metrics is a critical step in choosing the most suitable model for real-world applications.

---

## Evaluation Metrics Used

Since this was a classification problem, the following metrics were used to evaluate model performance:

Accuracy, Precision, Recall, and F1-score.

Based on these scores, the models were ranked.

---

### Accuracy

Accuracy measures the overall correctness of the model.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

It gives a general idea of performance but does not show how the model behaves for individual classes.

### Precision

Precision measures how many of the samples predicted for a class actually belong to that class.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

A high precision indicates that the model makes fewer false positive errors.

### Recall

Recall measures how many of the actual samples of a class were correctly identified.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

A high recall means the model is able to detect most of the instances of a class. This is important when missing an instance is costly.

### F1-score

F1-score is the harmonic mean of precision and recall.

$$
F1 = 2 \times \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

It provides a balanced measure when both false positives and false negatives need to be considered.

---

## Performance Comparison

After evaluating all five models on the same dataset, the results were tabulated for easier comparison.

![Performance Comparison](https://github.com/Vismaya0502/Marvel-Report-Images-2/blob/main/Screenshot%202026-02-23%20010556.png)

From the results, Logistic Regression and Support Vector Machine (SVM) had similar performance.

To obtain a much better estimate, cross-validation was performed.

Cross-validation showed that SVM achieved a higher mean accuracy and more consistent performance across different folds.

Hence, SVM was identified as the best-performing model for this task.

---

## Model Serialization using Pickle and Joblib

Pretrained models are stored in serialized(the conversion of a complex data object or structure (like a programming object, tree, or graph) into a flat, linear sequence of bytes or characters) form so that they can be reused without retraining.

Both pickle and joblib can be used for this purpose. Pickle is a general-purpose Python module for object serialization, while joblib is more efficient for machine learning models because it loads and saves models faster and it can handle large NumPy arrays efficiently  and it uses less memory.

---

[Notebook](http://localhost:8888/lab/tree/Evaluation_task.ipynb)

## Learning Outcomes

- This task showed how trained models can be saved and reused using serialized files, improving efficiency.
- When model performances are similar, cross-validation provides a much better and reliable estimate than train–test split.
- It also helped in understanding the concept of model serialization and its practical importance.

---

## Conclusion

In this task, multiple pretrained machine learning models were evaluated using the Iris dataset. The models were loaded using joblib and compared using accuracy, precision, recall, and F1-score.

Based on cross-validation results, SVM was identified as the best-performing and most consistent model.