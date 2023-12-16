<h1>Machine Learning Diabetes Diagnosis</h1>
This project focuses on exploring the application of hyperparameter tuning and preprocessing techniques in conjunction with random forest classification models for the diagnosis of diabetes. By utilizing various hypertuning and preprocessing methods, such as feature scaling, data centering, and resampling data, I aim to enhance the performance of the classification models in predicting diabetes.
<br>
<br>
<h2>Exploratory Analysis</h2>

<h5>Preview of the Data</h5>
<table>
  <tr>
    <th>gender</th>
    <th>age</th>
    <th>hypertension</th>
    <th>heart_disease</th>
    <th>smoking_history</th>
    <th>bmi</th>
    <th>hbA1c_level</th>
    <th>blood_glucose_level</th>
    <th>diabetes</th>
  </tr>
  <tr>
    <td>Female</td>
    <td>80</td>
    <td>0</td>
    <td>1</td>
    <td>never</td>
    <td>25.19</td>
    <td>6.6</td>
    <td>140</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Female</td>
    <td>54</td>
    <td>0</td>
    <td>0</td>
    <td>No Info</td>
    <td>27.32</td>
    <td>6.6</td>
    <td>80</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Male</td>
    <td>28</td>
    <td>0</td>
    <td>0</td>
    <td>never</td>
    <td>27.32</td>
    <td>5.7</td>
    <td>158</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Female</td>
    <td>36</td>
    <td>0</td>
    <td>0</td>
    <td>current</td>
    <td>23.45</td>
    <td>5.0</td>
    <td>155</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Male</td>
    <td>76</td>
    <td>1</td>
    <td>1</td>
    <td>current</td>
    <td>20.14</td>
    <td>4.8</td>
    <td>155</td>
    <td>0</td>
  </tr>
</table>

<h5>Summary of the Data</h5>

<table>
  <tr>
    <td>
      <table>
        <tr>
          <th>gender</th>
          <th>age</th>
          <th>hypertension</th>
        </tr>
        <tr>
          <td>Min.   :NA</td>
          <td>Min.   : 0.08</td>
          <td>Min.   :0.00000</td>
        </tr>
        <tr>
          <td>1st Qu.:NA</td>
          <td>1st Qu.:24.00</td>
          <td>1st Qu.:0.00000</td>
        </tr>
        <tr>
          <td>Median :NA</td>
          <td>Median :43.00</td>
          <td>Median :0.00000</td>
        </tr>
        <tr>
          <td>Mean   :NA</td>
          <td>Mean   :41.89</td>
          <td>Mean   :0.07485</td>
        </tr>
        <tr>
          <td>3rd Qu.:NA</td>
          <td>3rd Qu.:60.00</td>
          <td>3rd Qu.:0.00000</td>
        </tr>
        <tr>
          <td>Max.   :NA</td>
          <td>Max.   :80.00</td>
          <td>Max.   :1.00000</td>
        </tr>
      </table>
    </td>
    <td>
      <table>
        <tr>
          <th>heart_disease</th>
          <th>smoking_history</th>
          <th>bmi</th>
        </tr>
        <tr>
          <td>Min.   :0.00000</td>
          <td>Min.   :NA</td>
          <td>Min.   :10.01</td>
        </tr>
        <tr>
          <td>1st Qu.:0.00000</td>
          <td>1st Qu.:NA</td>
          <td>1st Qu.:23.63</td>
        </tr>
        <tr>
          <td>Median :0.00000</td>
          <td>Median :NA</td>
          <td>Median :27.32</td>
        </tr>
        <tr>
          <td>Mean   :0.03942</td>
          <td>Mean   :NA</td>
          <td>Mean   :27.32</td>
        </tr>
        <tr>
          <td>3rd Qu.:0.00000</td>
          <td>3rd Qu.:NA</td>
          <td>3rd Qu.:29.58</td>
        </tr>
        <tr>
          <td>Max.   :1.00000</td>
          <td>Max.   :NA</td>
          <td>Max.   :95.69</td>
        </tr>
      </table>
    </td>
    <td>
      <table>
        <tr>
          <th>hbA1c_level</th>
          <th>blood_glucose_level</th>
          <th>diabetes</th>
        </tr>
        <tr>
          <td>Min.   :3.500</td>
          <td>Min.   : 80.0</td>
          <td>Min.   :0.000</td>
        </tr>
        <tr>
          <td>1st Qu.:4.800</td>
          <td>1st Qu.:100.0</td>
          <td>1st Qu.:0.000</td>
        </tr>
        <tr>
          <td>Median :5.800</td>
          <td>Median :140.0</td>
          <td>Median :0.000</td>
        </tr>
        <tr>
          <td>Mean   :5.528</td>
          <td>Mean   :138.1</td>
          <td>Mean   :0.085</td>
        </tr>
        <tr>
          <td>3rd Qu.:6.200</td>
          <td>3rd Qu.:159.0</td>
          <td>3rd Qu.:0.000</td>
        </tr>
        <tr>
          <td>Max.   :9.000</td>
          <td>Max.   :300.0</td>
          <td>Max.   :1.000</td>
        </tr>
      </table>
    </td>
  </tr>
</table>
<br>

```ruby
head(data)
summary(data)
nrow(data)

#checking for duplicate rows
unique_rows <- unique(data)
n_unique_rows <- nrow(unique_rows)

#number of distinct values per column, checking binary data
distinct_counts <- sapply(data, function(x) length(unique(x)))

#checking for null values
any(is.na(data))

```

<br>
During my exploratory analysis, I focused on assessing the data quality and understanding the limitations of the dataset. A notable aspect was the absence of any null values which allowed me to retain all observations. The dataset encompassed various data types, including binary, categorical, dummy, and numeric variables. <br>
<br>

I uncovered a concerning observation regarding the blood_glucose_level and hbA1c_level variables. The blood_glucose_level variable exhibits a concentration of values around integer multiples of 5. This finding is troubling because blood glucose readings typically do not exhibit such grouping patterns. The hbA1c_level variable exhibits similar patterns as well. Consequently, this discovery raises doubts regarding the validity of the data and suggests the possibility of fabricated or inaccurate information. After all, this dataset was obtained from a popular online community of data scientists and machine learning engineers at kaggle.com (link at the bottom). Further investigation and validation of the data would be necessary before applying the results of this analysis to real world applications. For all intents and purposes, my analysis is focused on the application of hyperparameter tuning and preprocessing techniques of machine learning and not the integrity of the dataset.
<br>

<h2>Variables</h2>

1. **`Age`**: Age is a significant factor in determining the likelihood of developing diabetes. As individuals grow older, their risk increases due to factors like reduced physical activity, hormonal changes, and the possibility of developing other health issues that contribute to diabetes. Age is a numeric variable.

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/1a192d20-eee1-4828-8244-3140e2b6fc94" alt="pic" width="500" height="400">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/999573ed-0460-4005-9aa3-47fe63c3de44" width="500" height="400">
</div>

2. **`Gender`**: Gender also plays a role in diabetes risk, albeit with some variation (as we will come to find out). Gender is a categorical variable that is transformed into dummy variables for analysis (female, male, and other).

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/31f43889-b70f-4155-bdfa-19774fe32659" width="500" height="400">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/30d4a7a2-08b0-473d-9d8b-fa6c6f2f7fc7" width="500" height="400">
</div>

3. **`Body Mass Index (BMI)`:** BMI is a useful indicator for predicting diabetes risk. Excess body fat, particularly around the waist, can lead to insulin resistance and hinder the body's ability to regulate blood sugar levels. BMI is a numeric variable.

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/6c8db498-516d-4058-83a5-d2a87696cbc3" width="500" height="400">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/1a7adb64-7d50-46db-9c84-241875952772" width="500" height="400">
</div>

4. **`Hypertension`:** Hypertension (high blood pressure) raises the risk of developing diabetes, and vice versa. The presence of hypertension in any given observation is represented as 1, the absence thereof is represented as 0. Hypertension is a binary variable.

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/1b2c1ee5-18bd-4a4b-b0af-d85a1f13be47" width="500" height="400">

</div>

5. **`Heart Disease`:** Heart disease, including conditions such as coronary artery disease and heart failure, is associated with an increased risk of diabetes. The presence of heart disease in any given observation is represented as 1, the absense thereof is represented as 0. Heart disease is a binary variable.

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/284a22a2-96bd-4b26-8137-78ee55149d5c" width="500" height="400">

</div>

6. **`Smoking History`:** Smoking has been found to increase the risk of developing type 2 diabetes. Smoking can contribute to insulin resistance and impair glucose metabolism. Some studies suggest that quitting smoking can significantly reduce the risk of developing diabetes and its complications. Smoking history is a categorical variable that for the purposes of my analysis was changed to dummy variables where "No Info" and "never" = never_smoked, "ever", "former", and "non-current" = smoked_in_past, and "current" = currently_smokes. 

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/2443d593-a37b-476e-8a4d-930f36197e21" width="500" height="400">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/d486742a-ef11-47f8-9b6e-2fdca647794f" width="500" height="400">
</div>

7. **`HbA1c Level`:** HbA1c (glycated hemoglobin) is a measure of the average blood glucose level over the past 2-3 months. It provides information about long-term blood sugar control. Higher HbA1c levels indicate poorer glycemic control and are associated with an increased risk of developing diabetes and its complications. hbA1c_level is a numeric variable.

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/c0881102-037c-4d66-b2ba-459ed4eaef89" alt="pic" width="500" height="400">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/da18b94c-1f26-461c-a048-18d520f7d98a" width="500" height="400">
</div>

8. **`Blood Glucose Level`:** Blood glucose level refers to the amount of glucose (sugar) present in the blood at a given time. Elevated blood glucose levels, particularly in the fasting state or after consuming carbohydrates, can indicate impaired glucose regulation and increase the risk of developing diabetes. This is a numeric variable. 

<div style="text-align: center;">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/8551eaf9-f0ea-4e2a-887e-0a47b4268ce5" alt="pic" width="500" height="400">
  <img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/61e86b25-24f6-411f-9e54-85c27b390cbe" width="500" height="400">
</div>

9. **`Diabetes`:** The diabetes variable is the outcome variable which is used to train the classification models to determine whether the presence of diabetes can be inferred based on other variables. 

<img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/2ad983cf-0964-4bf9-9376-60189c969c32" width="500" height="400"> <br>
<br>
<br>

<h2>Multivariate Analysis</h2>

**`Scatterplot for age and bmi by diabetes`:** Represented by the red line is the line of best fit for people with diabetes. Similarly, represented by the gray line is the line of best fit for people without diabetes.
<br>
<img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/570e841f-9cba-4161-9252-0c14211107fc" width="500" height="400">
<br>
<br>

**`Scatterplot for age and hbA1c_level by diabetes`:** Notice the odd groupings across the hbA1c_level variable. 
<br>
<img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/98edf78a-632c-496b-b97d-4e4538239fd5" width="500" height="400">
<br>
<br>

**`Scatterplot for age and blood_glucose_level by diabetes`:** Notice the odd groupings across the blood_glucose_level variable. 
<br>
<img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/8f6ff0db-2e47-4876-aca0-1044ff6ec52f" width="500" height="400">
<br>
<br>

<h2>Pre-processing</h2>
In preprocessing the data, I convert categorical variables into dummy variables, scale and center the data, and partition 80% of the dataset for training and 20% for testing. I ultimately decided to center and scale numeric variables even though these transformations are not typically necessary for random forest classification models. I decided to make these transformations because many other machine learning models require scaling features and doing so will not affect the integrity of my model's predictions. Random forests combine multiple decision trees whereby each tree measures the probability that a random feature is indicative of some output. Each tree's decision is not affected by the scaling or variance of other trees. Seeds are placed throughout the script for reproducibility purposes.
<br>

```ruby
#convert categorical variables to dummy (not to be applied for exploratory analysis)
data$female <- ifelse(data$gender == "Female", 1, 0)
data$male <- ifelse(data$gender == "Male", 1, 0)
data$other <- ifelse(data$gender == "Other", 1, 0)
data$never_smoked <- ifelse(data$smoking_history %in% c("never", "No Info"), 1, 0)
data$smoked_in_past <- ifelse(data$smoking_history %in% c("ever", "former", "not current"), 1, 0)
data$currently_smokes <- ifelse(data$smoking_history %in% c("current"), 1, 0)

#delete columns
data$gender <- NULL
data$smoking_history <- NULL

#scale and center numeric variables
variables_to_scale <- data[, c('age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease')]
scaled_and_centered_variables <- as.data.frame(scale(variables_to_scale))
data[, c('age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease')] <- scaled_and_centered_variables

#convert diabetes to a factor
data.class(data$diabetes)
data$diabetes = as.factor(data$diabetes)

#seed for reproducibility
set.seed(1)

#data partitioning for random forest model
data_set_size = floor(nrow(data)*.8)
index <- sample(1:nrow(data), size = data_set_size)
training <- data[index,]
testing <- data[-index,]

```
<br>

<h2>Correlation</h2>
A correlation matrix was computed to examine the relationships between different variables in the dataset. Categorical variables were transformed into dummy variables. The correlation of diabetes with each variable indicates the strength and direction of their linear relationship. Positive correlations suggest that higher values of a variable are associated with a higher likelihood of diabetes, while negative correlations indicate an inverse relationship. Note that correlation does not imply causation, and additional analysis is needed to interpret casual relationships accurately.
<br>
<br>
<img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/73c38232-b307-4107-b482-6d5dd35d6138" width="500" height="400">
<br>
<br>
<table>
  <tr>
    <th>Variable</th>
    <th>Correlation</th>
  </tr>
  <tr>
    <td>diabetes</td>
    <td>1.00000000</td>
  </tr>
  <tr>
    <td>blood_glucose_level</td>
    <td>0.41955800</td>
  </tr>
  <tr>
    <td>hbA1c_level</td>
    <td>0.40066031</td>
  </tr>
  <tr>
    <td>age</td>
    <td>0.25800803</td>
  </tr>
  <tr>
    <td>bmi</td>
    <td>0.21435741</td>
  </tr>
  <tr>
    <td>hypertension</td>
    <td>0.19782325</td>
  </tr>
  <tr>
    <td>heart_disease</td>
    <td>0.17172685</td>
  </tr>
  <tr>
    <td>smoking_history</td>
    <td>0.07649274</td>
  </tr>
  <tr>
    <td>gender</td>
    <td>0.03724236</td>
  </tr>
</table>
<br>

```ruby
#finding non-numeric columns
non_numeric_vars <- !sapply(data, is.numeric)
non_numeric_columns <- names(data)[non_numeric_vars]
print(non_numeric_columns)

data <- as.data.frame(sapply(data, as.numeric))
str(data)

#color correlation matrix
correlation_matrix <- cor(data)
corrplot(correlation_matrix, method = "color", type = "upper", tl.cex = 0.7)

#correlation table diabetes with other variables
diabetes_correlation <- correlation_matrix[,"diabetes"]
diabetes_correlation

```

<br>
<h2>Predictive Analysis Using Random Forest</h2>
<h4>Model 1</h4>
<br>
Initially, I used training data to create a baseline random forest model and then applied resampled data to mitigate type two error. This baseline model (Model 1) serves as a reference point for evaluating the performance improvements achieved through subsequent transformations. Model 1 is tuned with the parameters mtry = 4 and ntree = 2001.
<br>

<h5>Model 1 Confusion Matrix</h5>
<table>
  <tr>
    <th rowspan="2">Actual</th>
    <th colspan="2">Predicted</th>
    <th rowspan="2">Class error</th>
  </tr>
  <tr>
    <th>0</th>
    <th>1</th>
  </tr>
  <tr>
    <th>0</th>
    <td>73072</td>
    <td>94</td>
    <td>0.00128475</td>
  </tr>
  <tr>
    <th>1</th>
    <td>2289</td>
    <td>4545</td>
    <td>0.33494293</td>
  </tr>
</table>
<br>

<h5>Model 1 Plot</h5>
<img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/a24e4f46-8482-40ba-b6fa-8825be299ef0" width="500" height="400"> <br>
<br>

In Model 1, the OOB estimate of 3% suggests that, on average, the model is expected to accurately predict 97% of the cases it encounters. However, there is a high class error rate of 33% for type two error (false negatives). This indicates that the model struggles to accurately diagnose diabetes when patients truly have diabetes. Given that approximately 91% of the observations are not diagnosed with diabetes, I suspected that addressing the majority-minority class variation would lower type two error. To test my theory, I applied both undersampling to the majority class and oversampling to the minority class of my next model to limit the bias of the model's prediction towards the majority class. I also employed a hyperparameter tuning algorithm on the mtry parameter and reduced the number of trees in the model from 2001 to 501.
<br>
<br>

```ruby
#seed for reproducibility
set.seed(1)

#random forest model without resampling to address the minority class
rf_model <- randomForest(diabetes ~ ., data = training, mtry = 4, ntree = 501, importance = TRUE)
rf_model

#See what the algorithm predicted against actual results
results_1 <- data.frame(testing$diabetes, predict(rf_model, testing[,1:8], type = "response"))
results_1

#plot shows number of trees can be truncated to 501 without much loss of predictive accuracy
plot(rf_model)

```

<br>

<h4>Model 2</h4>
<br>
Model 2 addresses the class imbalance through both oversampling and undersampling (N = 70,000). The result was an OOB estimate of 2.81% and a confusion matrix showing minimal class errors (Model 2 Confusion Matrix). Next, I hypertuned the mtry parameter. To do this, I applied the tuneRF algorithm which calculates the optimal number of mtry: number of variables randomly sampled as candidates at each split in the decision trees within the forest. The results of the tuneRF algorithm are shown in Model 2 Mtry Hypertuning Results.
<br>
<br>

<h5>Model 2 Confusion Matrix</h5>
<table>
  <tr>
    <th rowspan="2">Actual</th>
    <th colspan="2">Predicted</th>
    <th rowspan="2">Class error</th>
  </tr>
  <tr>
    <th>0</th>
    <th>1</th>
  </tr>
  <tr>
    <th>0</th>
    <td>41311</td>
    <td>853</td>
    <td>0.02023053</td>
  </tr>
  <tr>
    <th>1</th>
    <td>290</td>
    <td>27546</td>
    <td>0.01041816</td>
  </tr>
</table>

<br>

<h5>Model 2 Mtry Hypertuning Results</h5>
<img src="https://github.com/JonesCM4/MLforDiabetesDiagnosis/assets/126039483/a58ea7b7-3e36-49c6-be57-ae85560f40bc" width="500" height="400"> <br>
<br>

```ruby
#seed for reproducibility
set.seed(1)

#oversampling and undersampling for better sensitivity
table(training$diabetes)
both_sampling_data <- ovun.sample(diabetes ~ ., data = training, method = "both", p = 0.4, N = 70000)$data

table(both_sampling_data$diabetes)

both_sampling_data$diabetes <- as.factor(both_sampling_data$diabetes)

#seed for reproducibility
set.seed(1)

rf_model_2 <- randomForest(diabetes ~ ., data = both_sampling_data, mtry = 4, ntree = 501, importance = TRUE)
rf_model_2

#split data into independent and dependent variables
independent_variables <- both_sampling_data[, c("bmi", "age", "hypertension", "blood_glucose_level", "heart_disease", "HbA1c_level", "male", "female", "other", "never_smoked", "smoked_in_past", "currently_smokes")]
dependent_variable <- both_sampling_data$diabetes

#seed for reproducibility
set.seed(1)

#hypertune mtry
best_mtry <- tuneRF(x = independent_variables,
  y = dependent_variable,
  mtryStart = 5,
  ntreeTry = 20,
  stepFactor = 1.5,
  improve = 1e-5,
  trace = TRUE,
  plot = TRUE,
  doBest = TRUE,
  nodesize = 30,
  importance = TRUE)

print(best_mtry)
plot(best_mtry)

#see what algorithm 2 predicted against actual results
results_2 <- data.frame(testing$diabetes, predict(rf_model_2, testing[,1:8], type = "response"))
results_2

#plot model 2
plot(rf_model_w)

```

<br>

<h4>Model 3</h4>
<br>
In model 3, I applied the optimal mtry value of 5, as determined in the previous model. Model 3 evaluates model 2's performance using unseen testing data, the results are satisfactory. As a final step, I examine the importance of each variable on predicting the outcome.
<br>

<h5>Model 3 Confusion Matrix</h5>
<table>
  <tr>
    <th rowspan="2">Actual</th>
    <th colspan="2">Predicted</th>
    <th rowspan="2">Class error</th>
  </tr>
  <tr>
    <th>0</th>
    <th>1</th>
  </tr>
  <tr>
    <th>0</th>
    <td>17730</td>
    <td>390</td>
    <td>0.02152317</td>
  </tr>
  <tr>
    <th>1</th>
    <td>579</td>
    <td>1301</td>
    <td>0.30797872</td>
  </tr>
</table>

<h5>Model 3 Feature Importance</h5>
    <table>
  <tr>
    <th>Feature</th>
    <th> %IncMSE</th>
    <th>incNodePurity</th>
  </tr>
  <tr>
    <td>hbA1c_level</td>
    <td>219.5008523</td>
    <td>2416.24</td>
  </tr>
  <tr>
    <td>blood_glucose_level</td>
    <td>161.1516829</td>
    <td>1837.675</td>
  </tr>
  <tr>
    <td>age</td>
    <td>48.7531141</td>
    <td>372.8975</td>
  </tr>
  <tr>
    <td>bmi</td>
    <td>42.1644986</td>
    <td>434.3301</td>
  </tr>
  <tr>
    <td>hypertension</td>
    <td>23.6089331</td>
    <td>80.93079</td>
  </tr>
  <tr>
    <td>heart_disease</td>
    <td>19.857738</td>
    <td>43.06763</td>
  </tr>
  <tr>
    <td>never_smoked</td>
    <td>9.4755406</td>
    <td>21.41233</td>
  </tr>
  <tr>
    <td>currently_smokes</td>
    <td>8.0194219</td>
    <td>15.48485</td>
  </tr>
  <tr>
    <td>female</td>
    <td>7.1231758</td>
    <td>17.3637</td>
  </tr>
  <tr>
    <td>male</td>
    <td>6.1149731</td>
    <td>17.30025</td>
  </tr>
  <tr>
    <td>smoked_in_past</td>
    <td>14.7582729</td>
    <td>20.49736</td>
  </tr>
  <tr>
    <td>other</td>
    <td>0.9461699</td>
    <td>0.004573227</td>
  </tr>
</table>

<h5>Model 3 Feature Importance Explanation</h5>
The %IncMSE column shows the percentage increase in the mean squared error (MSE) when a variable is removed from model 3. For example, hbA1c_level has a %IncMSE value of 219.5008523, indicating that if the variable were to be removed from the model, the result would be a mean squared error increase of approximately 220%. IncNodePurity represents the increase in node purity associated with each variable. For example, the incNodePurity value for hbA1c_level is 2416.24 which can then be compared to the values of other variables to infer the variable's importance in the model. A high incNodePurity value indicates a variable's ability to significantly contribute to the purity of nodes in the decision tree. Interestingly, all the dummy variables in my model exhibit a percentage increase in the MSE of less than 10%. This suggests that these dummy variables may not be as relevant to the model compared to other factors.
<br>

```ruby
#apply model 2 to testing data
predicted_2 <- predict(rf_model_2, testing, type = "response")
predicted_2 <- factor(ifelse(predicted[, 2] > 0.5, "1", "0"), levels = c("0", "1"))
actual_3 <- factor(testing$diabetes, levels = c("0", "1"))

#model 3: model 2's performance on testing data
confusionMatrix(predicted_2, actual_3, positive = "1")

#model 2's feature importance
feature_importance_2 <- importance(rf_model_2)
feature_importance_2

```

<br>
<h2>Conclusion</h2>

This project explores hypertuning and preprocessing techniques in random forest models for the diagnosing of diabetes. I prioritized creating a model that made accurate predictions for the minority class (those who truly have diabetes). Through model optimization efforts, I was able to limit the type 2 error rate by approximately 3% while maintaining a relatively low type 1 error rate. The ultimate goal for this project was to gain some exposure to machine learning applications in R and maybe learn something new along the way. Although I addressed several challenges in optimizing random forest classification models, numerous others remain to be explored. Further analysis could involve omitting variables with low correlation to the outcome variable, additional parameter hypertuning on the node size or maximum tree depth, optimizing other resampling techniques, and testing other classification models for improved model fitting.

<br>
<h2>Helpful Links</h2>

I gained valuable insights that helped me complete this project thanks to the information made available in these resources:
<br>
- https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
- https://rpubs.com/phamdinhkhanh/389752
- https://www.blopig.com/blog/2017/04/a-very-basic-introduction-to-random-forests-using-r/
- https://medium.com/analytics-vidhya/a-random-forest-classifier-with-imbalanced-data-7ef4d9ebedb8
- https://towardsdatascience.com/what-is-a-decision-tree-22975f00f3e1
