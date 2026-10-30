# Assignment

## **1. Comedy show 🎭**

**Folder:** `session_3/1.decision_tree_comedy_show/`

A. Use the `sdv` package to generate synthetic data and augment the original dataset. Train a Decision Tree classifier on the augmented dataset.

```python
import sdv

metadata = sdv.metadata.Metadata.detect_from_dataframes(data={'comedian': df})
synthesizer = sdv.single_table.GaussianCopulaSynthesizer(metadata)
synthesizer.fit({'comedian': df})
synthetic_data = synthesizer.sample(num_rows=100)['comedian']
df_augmented = pd.concat([df, synthetic_data],ignore_index=True)
display(df_augmented)
```

B. Use `criterion='entropy'` parameter instead of `criterion='gini'` when creating a Decision Tree classifier. In two or three sentences, answer the following questions:

- What is the **Gini** method and how does it work?
- What are differences in the tree structure and feature importances between the model trained with **Gini** impurity and the model trained with **entropy**?

## **2.** Boston house-prices **🏘️**

A. Load Boston dataset from the `scikit-learn` packege and split it with `random_state=42`.

B. Train Random Forest regressors with `n_estimators=100`.

C. Evaluate your models using these metrics:

- MSE
- RMSE
- MAE
- MAPE
- R²

D. Plot actual versus predicted test values and add a diagonal line representing perfect predictions.

E. Train regressors with `n_estimators` values of `50`, `100`, and `200`. Create a table containing the number of trees, test RMSE, and test R² for each model.

F. Convert RMSE to dollars. Create a clearly labeled chart comparing RMSE across the three forests.

## **3. Titanic 🚢**

**Folder:** `session_3/4.random_forest_titanic/`

A. Use `feature_importances_` property of your trained Random Forest model object to identify which features are most important for predicting a comedian’s rank. Sort the features by importance and display them in a bar chart.

B. Calculate the median age from the training subset and use that same value to fill missing ages in both subsets. Evaluate survival predictions while learning the age-imputation value from training rows only.

C. Display a classification report and confusion matrix labeled `Did not survive` and `Survived`.

D. Explain the false positives and false negatives when survival is the positive class, and identify which class has higher recall.

E. Train four models with `max_depth` values of `1`, `2`, `4`, and `8`, using `random_state=42` for each. Create a comparison table containing maximum depth, actual tree depth, number of leaves, and training accuracy. Investigate how limiting a Decision Tree’s depth changes its rules and training performance.

F. Save your test predictions as a CSV file like the `gender_submission.csv` sample file, with `PassengerId` and `Survived` columns. Submit the CSV file to **Kaggle** and report your test accuracy.

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## **4. Give Me Some Credit 💸**

**Folder:** `session_3/5.give_me_some_credit/`

Complete the preparation workflow, then investigate whether the Random Forest detects delinquency beyond achieving high overall accuracy.

Your notebooks should:

A. Export the prepared `train.csv` and `test.csv` files without a DataFrame index.

B. Report precision, recall, F1-score, and support for delinquency (`SeriousDlqin = 1`), alongside a clearly labeled confusion matrix. Explain why accuracy alone can be misleading.

C. Increase accuracy by balancing data, Remove outliers from train set, Change classifier model or tuning hyperparameters, and report the best model and hyperparameter values and their corresponding accuracy.

[SMOTENC — Version 0.15.dev0](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTENC.html)

## **5. Women vs Men 👫**

![](https://4183060045-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fu9ESMkINnUK9Z0nC4FPH%2Fuploads%2Fj7lpeCaBOul6WoqFmmH2%2Fimage.png?alt=media&token=4b9d18a1-f46c-4081-9a0e-2d456fed027d)

A. Download ANSUR II dataset from kaggle or openlab website.

[ANSUR II](https://www.openlab.psu.edu/ansur2)

[ANSUR II](https://www.kaggle.com/datasets/seshadrikolluri/ansur-ii)

B. Preprocess and clean dateset.

C. Show heights for women and men on same plot. A. Why is the data of men higher than the data of women? ‌B. Why is the data of men more right than the data of women?

D. Split dataset to train and test datasets (%80 for train and %20 for test):

```python
from sklearn.model_selection import train_test_split
```

E. Create and fit an ensemble classifier model on the train dataset. Use Bootstrap Aggregation (bagging) ensembling method to improve the accuracy and performance of the Decision Tree classifier.

<aside>
🚫

DON'T USE `RandomForestClassifier`. Instead, use `BaggingClassifier` with `DecisionTreeClassifier` as the base estimator. Compare the performance of the bagged model with the original Decision Tree classifier.

[Python Machine Learning - Bootstrap Aggregation (Bagging)](https://www.w3schools.com/python/python_ml_bagging.asp)

</aside>

F. Evaluate your model on the test dataset and report the accuracy.

G. Calculate confusion matrix for test dataset.

## **Submission checklist**

Before submitting, make sure that:

- [ ]  Every chart has a title, axis labels, and readable labels or legends where relevant.
- [ ]  Calculations are visible in code and results are explained in Markdown.
- [ ]  Training scores and test scores are clearly distinguished.
- [ ]  Learned preprocessing statistics come from training rows only, and test rows are excluded from model fitting.
- [ ]  Random seeds are fixed for reproducible comparisons.
- [ ]  The original datasets remain unchanged; only the prepared credit exports are regenerated.
