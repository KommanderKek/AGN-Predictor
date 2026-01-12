from astropy.table import Table
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

file_name = input("Enter File Name: ")
fermi_swift = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\" + file_name, hdu = 1))

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

scores = []
for n_estimators in range(200, 1200, 200):
    for min_samples_split in range(2, 6):
        for max_depth in range(1, 21):
            if max_depth == 1:
                max_depth = None
            rfc = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split, max_depth = max_depth, class_weight = "balanced_subsample", n_jobs = 50, random_state = 42)
            start_time = time.perf_counter()
            cv_results = cross_validate(estimator = rfc, X = df_data, y = label_vals, cv = 5, scoring = "roc_auc")
            end_time = time.perf_counter()
            time_duration = end_time - start_time
            results = (n_estimators, min_samples_split, max_depth, time_duration, np.average(cv_results["test_score"]))
            print(results)
            scores.append(results)
print()
for score in scores:
    print(f"{score[0]}, {score[1]}, {score[2]}, {score[3]}, {score[4]}")