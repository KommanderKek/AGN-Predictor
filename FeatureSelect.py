from astropy.table import Table
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

fermi_swift = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\fermi_swift.fits", hdu = 1))

df_data = fermi_swift.to_pandas()
df_data = df_data.sample(frac = 1).reset_index(drop=True)
df_data = df_data[df_data["CLASS"] != b"UNK"]

label_vals = []
for label in df_data["CLASS"]:
    if label == b"AGN":
        label_vals.append(1)
    else:
        label_vals.append(0)

df_data.drop(columns = ['CLASS'], inplace = True)

removed_columns = []
test_scores = []
importance_scores = []

n_estimators = 400
min_samples_split = 2
max_depth = 14

# function to remove columns and analyze scores
def remove_unimportant_col():
    #Feature correlation
    print("correlating features...")
    feature_correlation = {}
    for i in range(len(df_data.columns)):
        feature_correlation.update({df_data.columns[i]: False})
    for i in range(len(df_data.columns)):
        if feature_correlation.get(df_data.columns[i]):
            continue
        column1 = df_data[df_data.columns[i]] 
        for j in range(i + 1, len(df_data.columns)):
            column2 = df_data[df_data.columns[j]]
            coeff = spearmanr(column1, column2).statistic
            if coeff >= 0.95:
                feature_correlation.update({df_data.columns[i]: True})
                feature_correlation.update({df_data.columns[j]: True})

    print("n_columns:", len(df_data.columns))

    rfc = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split, max_depth = max_depth, class_weight = "balanced_subsample", n_jobs = 50, criterion = "entropy")
    cv_results = cross_validate(estimator = rfc, X = df_data, y = label_vals, cv = 5, scoring = "roc_auc", return_estimator = True)
    
    # test accuracy
    score = np.average(cv_results['test_score'])
    print("test score: ", score)
    test_scores.append(score)

    # calculate average feature importances
    importances_avg = np.mean(np.array([cv_results["estimator"][i].feature_importances_ for i in range(5)]), axis = 0)
    df_feature_importance = pd.DataFrame(data = importances_avg, index = df_data.columns, columns = ['Importance'])
    df_feature_importance.sort_values(by = ['Importance'], ascending = True, inplace = True)

    # remove correlated column of least importance
    print("removing correlated columns...")
    dropped = False
    for index in range(len(df_feature_importance)):
        col_name = df_feature_importance.index[index]
        if feature_correlation.get(col_name):
            df_data.drop(columns = col_name, inplace = True)
            removed_columns.append(col_name)
            print("dropped correlated", col_name + ": Importance =", df_feature_importance['Importance'].iloc[index])
            dropped = True
            break
    if not dropped:
        first_importance = df_feature_importance['Importance'].iloc[0]
        col_name = df_feature_importance.index[0]
        df_data.drop(columns = col_name, inplace = True)
        removed_columns.append(col_name)
        print("dropped uncorrelated", col_name + ": Importance =", first_importance)

num_columns_remove = len(df_data.columns)
for i in range(num_columns_remove):
    print()
    print("Iteration:", i + 1)
    remove_unimportant_col()
plt.plot(range(num_columns_remove), test_scores)
plt.title("10-Fold Cross Validation Test Scores")
plt.xlabel("Number of Columns Removed")
plt.yticks(np.arange(0.90, 0.96, step = 0.01))
plt.show()

num_to_remove = input("How many columns to remove: ")
fermi_swift.remove_columns(removed_columns[:len(removed_columns) - num_to_remove])
fermi_swift.write(os.path.dirname(os.getcwd()) + "\\Catalogues\\fermi_swift_final.fits", format = "fits", overwrite = True)