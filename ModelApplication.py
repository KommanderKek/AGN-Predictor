from astropy.table import Table
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

fermi_swift = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\fermi_swift_final.fits", hdu = 1))

df_data = fermi_swift.to_pandas()
df_train = df_data[df_data["CLASS"] != b"UNK"]
df_app = df_data[df_data["CLASS"] == b"UNK"]

print("AGN", sum(df_train["CLASS"] == b"AGN"))
print("NAG", sum(df_train["CLASS"] == b"NAG"))
print("UNK", df_app.shape[0])

label_vals = []
for label in df_train["CLASS"]:
    if label == b"AGN":
        label_vals.append(1)
    else:
        label_vals.append(0)

df_train.drop(columns = ['CLASS'], inplace = True)
df_app.drop(columns = ['CLASS'], inplace = True)

n_estimators = 600
min_samples_split = 4
max_depth = None

rfc = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split, max_depth = max_depth, class_weight = "balanced_subsample", n_jobs = 50)
rfc.fit(X = df_train, y = label_vals)
results = rfc.predict_proba(df_app)

high_conf_agn = [results[i][1] >= 0.7 for i in range(len(results))]
low_conf_agn = [results[i][1] >= 0.5 and results[i][1] < 0.7 for i in range(len(results))]
high_conf_nag = [results[i][0] >= 0.7 for i in range(len(results))]
low_conf_nag = [results[i][0] >= 0.5 and results[i][0] < 0.7 for i in range(len(results))]

fermi_swift = fermi_swift[fermi_swift["CLASS"] == "UNK"]
high_conf_agn = fermi_swift[high_conf_agn]
low_conf_agn = fermi_swift[low_conf_agn]
high_conf_nag = fermi_swift[high_conf_nag]
low_conf_nag = fermi_swift[low_conf_nag]

fermi_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\LAT\\fermi_raw.fits", hdu = 1))

high_conf_agn = fermi_data[[fermi_data["Pivot_Energy"][i] in high_conf_agn["Pivot_Energy"] for i in range(len(fermi_data))]]
low_conf_agn = fermi_data[[fermi_data["Pivot_Energy"][i] in low_conf_agn["Pivot_Energy"] for i in range(len(fermi_data))]]
high_conf_nag = fermi_data[[fermi_data["Pivot_Energy"][i] in high_conf_nag["Pivot_Energy"] for i in range(len(fermi_data))]]
low_conf_nag = fermi_data[[fermi_data["Pivot_Energy"][i] in low_conf_nag["Pivot_Energy"] for i in range(len(fermi_data))]]

print(high_conf_agn)
print(low_conf_agn)
print(high_conf_nag)
print(low_conf_nag)

importances = rfc.feature_importances_
plt.title("Feature Importances")
plt.xlabel("Feature Name")
plt.ylabel("Importance Fraction")
plt.plot(df_train.columns, importances)
plt.show()