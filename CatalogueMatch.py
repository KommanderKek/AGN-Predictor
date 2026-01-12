from astropy.table import Table, Column, MaskedColumn
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from collections import defaultdict
import os
import numpy as np

def inside_ellipse(src_ra, src_dec,
                   ra, dec, major_95, minor_95, pos_ang):
    dx = (src_ra - ra) * np.cos(np.deg2rad(dec))
    dy = (src_dec - dec)

    cos_t = np.cos(np.deg2rad(pos_ang))
    sin_t = np.sin(np.deg2rad(pos_ang))
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    return (x_rot / major_95)**2 + (y_rot / minor_95)**2 <= 1.0
    
def match_counterparts(counterpart_dir, ra_name, dec_name, file_name):
    # read data
    fermi_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\LAT\\fermi_raw.fits", hdu = 1))
    cpart_data = Table(Table.read(os.path.dirname(os.getcwd()) + counterpart_dir, hdu = 1))

    # get source coordinates
    fermi_coords = SkyCoord(fermi_data["RAJ2000"], fermi_data["DEJ2000"], unit = "deg")
    cpart_coords = SkyCoord(cpart_data[ra_name], cpart_data[dec_name], unit = "deg")

    # convert sky areas to pixels
    hp = HEALPix(nside = 32, order = "ring")
    fermi_pix = hp.lonlat_to_healpix(fermi_coords.ra, fermi_coords.dec)
    cpart_pix = hp.lonlat_to_healpix(cpart_coords.ra, cpart_coords.dec)

    # convert to dictionary for fast lookup
    cpart_dict = defaultdict(list)
    for i, pix in enumerate(cpart_pix):
        cpart_dict[pix].append(i)

    # match sources
    cpart_matches = []
    for fermi_idx, pix in enumerate(fermi_pix):
        cparts = cpart_dict.get(pix, [])
        if len(cparts) == 0:
            continue
        
        ra0 = fermi_data["RAJ2000"][fermi_idx]
        dec0 = fermi_data["DEJ2000"][fermi_idx]
        major = fermi_data["Conf_95_SemiMajor"][fermi_idx]
        minor = fermi_data["Conf_95_SemiMinor"][fermi_idx]
        pa = fermi_data["Conf_95_PosAng"][fermi_idx]

        for cpart_idx in cparts:
            ra_c = cpart_data[ra_name][cpart_idx]
            dec_c = cpart_data[dec_name][cpart_idx]
            if inside_ellipse(ra_c, dec_c, ra0, dec0, major, minor, pa):
                cpart_matches.append((fermi_idx, cpart_idx))
    
    match_table = Table(rows = cpart_matches,
                        names = ["FERMI_INDEX", "SWIFT_INDEX"])
    print("Swift matches", len(match_table))
    print("Fermi matches", len(match_table.to_pandas()["FERMI_INDEX"].unique()))
    print()
    print(match_table)
    match_table.write(os.path.dirname(os.getcwd()) + "\\Catalogues\\" + file_name, format = "fits", overwrite = True)

# select fermi sources
fermi_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\LAT\\gll_psc_v35.fit", hdu = 1))

print("fermi before:", len(fermi_data))
fermi_data = fermi_data[~fermi_data["Signif_Avg"].mask]
fermi_data = fermi_data[fermi_data["Signif_Avg"] >= 5]
fermi_data = fermi_data[fermi_data["ROI_num"] >= 0]
fermi_data["Extended_Source_Name"] = [str(fermi_data[index]["Extended_Source_Name"]).strip() for index in range(len(fermi_data))]
fermi_data = fermi_data[fermi_data["Extended_Source_Name"] == ""]
fermi_data.add_column(Column(data = range(len(fermi_data)), name = "INDEX"))
print("fermi after:", len(fermi_data))

fermi_data.write(os.path.dirname(os.getcwd()) + "\\Catalogues\\LAT\\fermi_raw.fits", format = "fits", overwrite = True)

# select swift sources and match to fermi
swift_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\Swift\\2SXPS_Sources_clean.fits", hdu = 1))

print("swift before", len(swift_data))
swift_data = swift_data[swift_data["DetFlag"] == 0]
swift_data = swift_data[swift_data["FieldFlag"] == 0]
swift_data = swift_data[swift_data["OpticalLoadingWarning"] == 0]
swift_data = swift_data[swift_data["StrayLightWarning"] == 0]
swift_data = swift_data[swift_data["NearBrightSourceWarning"] == 0]
swift_data = swift_data[swift_data["IsPotentialAlias"] == 0]
swift_data = swift_data[swift_data["NearestNeighbour"] > swift_data["Err90"]]
swift_data = swift_data[swift_data["Exposure"] > 500]
swift_data.add_column(Column(data = range(len(swift_data)), name = "INDEX"))
print("swift after", len(swift_data))

swift_data.write(os.path.dirname(os.getcwd()) + "\\Catalogues\\Swift\\swift_raw.fits", format = "fits", overwrite = True)

match_counterparts("\\Catalogues\\Swift\\swift_raw.fits", "RA", "Decl", "swift_match.fits")