from astropy.table import Table, MaskedColumn, Column
from numpy.ma.core import MaskedConstant
import numpy as np
import os

fermi_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\LAT\\fermi_raw.fits", hdu = 1))

# remove unneeded columns
fermi_cols_to_remove = ["Source_Name", "DataRelease", "ROI_num", "Extended_Source_Name",
                        "Flags", "ASSOC_4FGL", "ASSOC_FGL", "ASSOC_FHL", "ASSOC_GAM1",
                        "ASSOC_GAM2", "ASSOC_GAM3", "ASSOC_TEV", "TEVCAT_FLAG",
                        "ASSOC1", "ASSOC2", "ASSOC_PROB_BAY", "ASSOC_PROB_LR",
                        "Unc_Counterpart",
                        "RAJ2000", "DEJ2000", "GLON", "GLAT", "Signif_Avg",
                        "Sqrt_TS_Band", "Signif_Peak", "Sqrt_TS_History", "Npred",
                        "Time_Peak", "Peak_Interval", "Flux_History", "Unc_Flux_History",
                        "Flux_Band", "Unc_Flux_Band", "nuFnu_Band"]

fermi_data.remove_columns(fermi_cols_to_remove)

'''# split arrays into separate columns
array_columns = ["Flux_Band", "nuFnu_Band", "Sqrt_TS_Band",
               "Flux_History", "Sqrt_TS_History"]

for col_name in array_columns:
    column = fermi_data[col_name]
    for index in range(len(fermi_data[col_name][0])):
        values = [row[index] for row in column]
        fermi_data.add_column(col = values, name = col_name + "_" + chr(index + 97))

fermi_data.remove_columns(array_columns)'''

# label encoding
fermi_data.add_column(col = [0] * len(fermi_data), name = "Spectrum")
fermi_data["SpectrumType"] = [str(fermi_data[index]["SpectrumType"]).strip() for index in range(len(fermi_data))]

for row in range(len(fermi_data)):
    if fermi_data[row]["SpectrumType"] == "LogParabola":
        fermi_data[row]["Spectrum"] = 1
    elif fermi_data[row]["SpectrumType"] == "PLSuperExpCutoff":
        fermi_data[row]['Spectrum'] = 2
fermi_data.remove_column("SpectrumType")

# set masks to median values
for col_name in fermi_data.colnames:
    if not isinstance(fermi_data[col_name], MaskedColumn):
        continue
    median = np.median(fermi_data[col_name][~fermi_data[col_name].mask])
    fermi_data[col_name][fermi_data[col_name].mask] = median

# set infinites to median values
for col_name in ["Flux_Peak", "Unc_Flux_Peak"]:
    fermi_data[col_name][fermi_data[col_name] == float("inf")] = float("nan")
    fermi_data[col_name][fermi_data[col_name] == float("-inf")] = float("nan")
    median = np.nanmedian(fermi_data[col_name])
    fermi_data[col_name][fermi_data[col_name] == float("inf")] = median
    fermi_data[col_name][fermi_data[col_name] == float("-inf")] = median

# set classes for labels
fermi_data.remove_column('CLASS2')
fermi_data.rename_column('CLASS1', 'CLASS')

fermi_data['CLASS'] = [str(fermi_data[index]['CLASS']).strip().upper() for index in range(len(fermi_data))]
fermi_data['CLASS'][fermi_data['CLASS'] == ""] = "UNK"

# training sample for agn / non-agn
agn_columns = ["AGN", "BCU", "BLL", "CSS", "FSRQ", "SEY", "SSRQ"]
for index in range(len(fermi_data)):
    if fermi_data['CLASS'][index] in agn_columns:
        fermi_data['CLASS'][index] = "AGN"
    elif fermi_data['CLASS'][index] != "UNK":
        fermi_data['CLASS'][index] = "NAG"

# print and save results
print("rows", len(fermi_data))
print("cols", len(fermi_data.colnames))
print("AGN", sum(fermi_data['CLASS'] == "AGN"))
print("NAG", sum(fermi_data['CLASS'] == "NAG"))
print("UNK", sum(fermi_data['CLASS'] == "UNK"))
fermi_data.write(os.path.dirname(os.getcwd()) + "\\Catalogues\\LAT\\fermi_data.fits", format = "fits", overwrite = True)