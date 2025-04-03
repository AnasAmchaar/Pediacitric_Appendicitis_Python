# Binary variables to predict
__________________________
> - ### Diagnosis ( appendicitis vs no-append )
> - ### Management ( conservative (bla 3amaliya) vs surgical )
> - ### severity ( kayn severity (moda3afat) vs makaynsh )

# Preprocessing
____________________
> - ### kNN for missing values

# EDA
_______________________
> - Table 2 ( Sginficance of a variable suing p-value)


# Used ML Models
________________________
> - ### Logistic Regression ( LR )
> - ### Random forest ( RL )
> - ### Generalized Boosted Regression (GBM)

# Dataset Types
____________________________
> - Full set of 38 predictor variables (dataset kamla)
> - Without US data (“US-free” : bla Ultra-Sound cuz s3iba tl9aha shi mrat fsbita)
> - Without the “peritonitis/abdominal guarding” variable (Doctor khasso ykon experienced especially for young kids)
> - Without US data or the “peritonitis/abdominal guarding” variable


# Evaluation Metrics
_______________________
## AUROC
> **AUROC** is a way to evaluate how well a model can **distinguish between two classes** (like "yes" vs. "no" or "positive" vs. "negative").
> #### **What Does AUROC Tell You?**
>- **AUROC = 1**: Perfect model. The model can perfectly distinguish between positive and negative cases.
>- **AUROC = 0.5**: The model performs no better than random chance (like flipping a coin).
>- **AUROC < 0.5**: The model is worse than random. It might be predicting the wrong class most of the time.
## AUPR
>**AUPR** is another way to assess a model's performance, but it focuses more on **precision** and **recall**, which are important in cases where one class is much rarer than the other (e.g., detecting rare diseases).
>- Precision = True Positives / (True Positives + False Positives)
>- **Recall = True Positives / (True Positives + False Negatives)**
>- **AUPR = 1**: Perfect model. The model has perfect precision and recall.
>- **AUPR = 0**: The model does terribly at distinguishing the positive class.
## Difference
>- **AUROC** is a good measure when you have a **balanced dataset**, meaning the positive and negative classes are roughly equal in size.
>- **AUPR** is more useful when you have an **imbalanced dataset** (e.g., predicting rare diseases, fraud detection), because it focuses on the **positive class**
# Variable Selection
____________________________
	In a clinical setting, variables can be systemically missing at test
	time. We therefore also examined the importance of predictor
	variables in case the number of predictors used by classifiers
	could be reduced without compromising their performance. Both
	RF and GBM provide measures of variable importance
>## Method Used 
>The procedure can be summarized as
follows. For number of predictors q from 1 to 38, repeat:
>1. Train full RF model Mfull (all predictor variables included) on
the train set. Retrieve variable importance values.
>2. Train RF model Mq based on q predictors with the highest
importance values, on the train set.
>3. Evaluate AUROC and AUPR of Mq on the test set.
>4. Repeat steps 1-3 for all 10 folds in CV.
>- Finally; For q from 1 through 38, we trained
random forest classifiers on 300 bootstrap resamples of the data
and counted how many times each predictor was among the
q most important variables

## Key Insights
```
- "We compared model performance using two-sided 10-fold cross-
validated paired t-tests at a significance level α = 0.05 (28). In
addition to AUROC and AUPR, sensitivity, specificity, negative
and postive predictive values of the classifiers were evaluated."
- Alvarado Score (AS) and Paediatric Appendicitis Score (PAS) are metrics doctors use to determine wether patient has appendicitis or not
```


makmltsh mora Discussion walakin mnin 3titha LLMs kay9olo makaynsh something useful
