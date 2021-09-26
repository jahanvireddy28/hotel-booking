#Import Libraries

import numpy as np
import pandas as pd
from category_encoders import *
from sklearn.compose import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import *
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score,  precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split

#Load data
booking_data = pd.read_csv("hotel_bookings.csv")

#Extract Target
y = booking_data["is_canceled"] # extract target
X = booking_data.drop(["is_canceled"],axis=1)

cat_columns = X.dtypes==object
con_columns = ~cat_columns
Identify columns to drop

cat_columns [["company", "agent", "reservation_status", "reservation_status_date"]] = False
con_columns [["company", "agent", "reservation_status", "reservation_status_date"]] = False

#Split the data into train and test datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
Build categorical and continuous pipelines

cat_pipe = Pipeline([("cat_imputer", SimpleImputer(missing_values=np.nan,
                                              strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore"))])

con_pipe = Pipeline([("con_imputer", SimpleImputer(missing_values=np.nan,
                                                  strategy="median")),
                    ("scaler", StandardScaler())])

preprocessing = ColumnTransformer([("categorical", cat_pipe, cat_columns),
                                  ("continuous", con_pipe, con_columns)],
                                 remainder="drop")
#Fit models
#Build pipeline

pipe = Pipeline([("preprocessing", preprocessing),
                ("rf", RandomForestClassifier())])

rf_hyperparams = {"rf__n_estimators": [10,20,50], # number of trees used in random forest, very high values could lead to overfitting
                 "rf__max_depth": [5, 10,15], # max depth of each tree, if the depth is too low, the accuracy is poor
                 "rf__criterion": ["gini", "entropy"], # to check whether impurity or information gain is the best way to split
                 "rf__min_samples_leaf": [3,5,10], # minimum samples beyond which a node cannot be split, higher values imply more generality
                 "rf__max_features": ["sqrt", "log2"], # to check what is the best way limit the number of features to each tree
                 "rf__bootstrap": [True, False]} # to check whether bagging and aggregating results in a better model

f1_wtd = make_scorer(f1_score, average="weighted")
pr_wtd = make_scorer(precision_score, average="weighted")
bal_acc_score = make_scorer(balanced_accuracy_score)
recall_wtd = make_scorer(recall_score, average = "weighted")

scoring_dict = {"bal_acc_score": bal_acc_score,
            "f1_wtd": f1_wtd,
            "pr_wtd": pr_wtd,
            "recall_wtd": recall_wtd}
#Fit models

rscv = RandomizedSearchCV(estimator=pipe,
                    param_distributions=rf_hyperparams,
                    n_iter = 10,
                    scoring = scoring_dict,
                    refit = "bal_acc_score",
                    n_jobs = -1,
                    cv = 5,
                    random_state=42)
result = rscv.fit(X_train, y_train)

#Print best hyperparameters

print(result.best_params_)
print(result.best_score_)
{'rf__n_estimators': 20, 'rf__min_samples_leaf': 10, 'rf__max_features': 'sqrt', 'rf__max_depth': 15, 'rf__criterion': 'entropy', 'rf__bootstrap': False}


pipe = Pipeline([("preprocessing", preprocessing),("log_reg", LogisticRegression())])

log_reg_hyperparams = {"log_reg__penalty": ['l1','l2'], # to check which penalty is better suited for the dataset
                      "log_reg__fit_intercept": [True],
                      "log_reg__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], # to see the impact of strength of regularization
                      "log_reg__class_weight": ["balanced"], # to account for imbalance in the dataset
                      }

rscv = RandomizedSearchCV(estimator=pipe,
                    param_distributions=log_reg_hyperparams,
                    n_iter = 10,
                    scoring = scoring_dict,
                    refit = "bal_acc_score",
                    n_jobs = -1,
                    cv = 5,
                    random_state=42)

result = rscv.fit(X_train, y_train)

print(result.best_params_)
print(result.best_score_)

#Fit the final model on training data

pipe = result.best_estimator_
model = pipe.fit(X_train, y_train)

# Print model hyper parameters
model.get_params
Out[ ]:
<bound method Pipeline.get_params of Pipeline(steps=[('preprocessing',
                 ColumnTransformer(transformers=[('categorical',
                                                  Pipeline(steps=[('cat_imputer',
                                                                   SimpleImputer(strategy='most_frequent')),
                                                                  ('ohe',
                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                dtype: bool)])),
                ('log_reg',
                 LogisticRegression(C=1000, class_weight='balanced'))])>
#Get predictions for the test data set

y_pred = model.predict(X_test)
#Get the metrics for final model
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred): .4f}")
print(f"Precision score: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall score: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 score: {f1_score(y_test, y_pred, average='weighted'):.4f}")