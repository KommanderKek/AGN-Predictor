from astropy.table import Table, hstack
import numpy as np
from numpy.ma.core import MaskedConstant
import os

def distance(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)
    dx = np.sin(dec1) * np.sin(dec2)
    dy = np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.arccos(dx + dy)

# combine fermi and swift
# load data
fermi_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\LAT\\fermi_data.fits", hdu = 1))
swift_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\Swift\\swift_data.fits", hdu = 1))
swift_match = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\swift_match.fits", hdu = 1))

# remove fermi sources with no swift matches
rows_to_remove = [fermi_data[row]["INDEX"] not in swift_match["FERMI_INDEX"] for row in range(len(fermi_data))]
fermi_data.remove_rows(rows_to_remove)

# prepare table to append to the fermi sources
swift_feats = swift_data.copy()
swift_feats.remove_rows(range(len(swift_feats)))

# get counterpart features
for row in range(len(fermi_data)):
    index = fermi_data[row]["INDEX"]
    cparts_idx = swift_match["SWIFT_INDEX"][swift_match["FERMI_INDEX"] == index]
    cparts = swift_data[cparts_idx]
    cpart = cparts[0]
    if len(cparts) > 1:
        cparts_ra = cparts["RA"]
        cparts_dec = cparts["Decl"]
        assoc_ra = fermi_data[row]["RA_Counterpart"]
        assoc_dec = fermi_data[row]["DEC_Counterpart"]
        distances = [distance(cparts_ra[i], cparts_dec[i], assoc_ra, assoc_dec) for i in range(len(cparts))]
        min_dist_idx = distances.index(min(distances))
        cpart = cparts[min_dist_idx]
    swift_feats.add_row(cpart)

# remove no longer needed columns
swift_feats.remove_columns(["INDEX", "RA", "Decl"])
fermi_data.remove_columns(["INDEX", "RA_Counterpart", "DEC_Counterpart"])

# append fermi and swift together
fermi_swift = hstack([fermi_data, swift_feats])

print("rows", len(fermi_swift))
print("cols", len(fermi_swift.colnames))
print("AGN", sum(fermi_swift["CLASS"] == "AGN"))
print("NAG", sum(fermi_swift["CLASS"] == "NAG"))
print("UNK", sum(fermi_swift["CLASS"] == "UNK"))

fermi_swift.write(os.path.dirname(os.getcwd()) + "\\Catalogues\\fermi_swift.fits", format = "fits", overwrite = True)