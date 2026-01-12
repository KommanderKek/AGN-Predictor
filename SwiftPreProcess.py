from astropy.table import Table, MaskedColumn, Column
from numpy.ma.core import MaskedConstant
import numpy as np
import os

swift_data = Table(Table.read(os.path.dirname(os.getcwd()) + "\\Catalogues\\Swift\\swift_raw.fits", hdu = 1))

# remove unneeded columns
columns_to_remove = ["2SXPS_ID", "IAUName", "AstromType", "NearestNeighbour",
                    "OrigErr90", "NearestOKNeighbour", "BestDetectionID",
                    "FirstObsDate", "LastObsDate", "FirstObsMET", "LastObsMET",
                    "FirstDetDate", "LastDetDate", "FirstDetMET", "LastDetMET",
                    "FirstBlindDetDate", "LastBlindDetDate", "FirstBlindDetMET",
                    "LastBlindDetMET", "NumObs", "NumBlindDetObs", "NumDetObs",
                    "DetFlag", "FieldFlag", "DetFlag_band0", "DetFlag_band1",
                    "DetFlag_band2", "l", "b", "MeanOffAxisAngle",
                    "DetFlag_band3", "OpticalLoadingWarning", "StrayLightWarning",
                    "NearBrightSourceWarning", "IsPotentialAlias",
                    "PotentialAliasList", "WhichPow", "WhichAPEC", "HasSpec",
                    "NumExternalMatches", "NumExternalMatches_slim", "MatchInROSHRI",
                    "MatchIn2RXS", "MatchIn3XMMDR8", "MatchIn3XMM_Stack",
                    "MatchInXMMSL2", "MatchInSwiftFT", "MatchIn1SWXRT",
                    "MatchInXRTGRB", "MatchInSDSSQSO", "MatchIn2MASS",
                    "MatchInUSNOB1", "MatchIn2CSC", "MatchIn1SXPS",
                    "MatchInALLWISE", "RateCF_band0", "RateCF_band1", "RateCF_band2",
                    "RateCF_band3", "UL_band0", "UL_band1", "UL_band2", "UL_band3",
                    "GalacticNH", "l", "b",
                    "FixedPowECFO", "FixedPowECFU", "FixedPowFlux", "FixedPowFlux_pos",
                    "FixedPowFlux_neg", "FixedPowUnabsFlux", "FixedPowUnabsFlux_pos",
                    "FixedPowUnabsFlux_neg", "FixedAPECECFO", "FixedAPECECFU",
                    "FixedAPECFlux", "FixedAPECFlux_pos", "FixedAPECFlux_neg",
                    "FixedAPECUnabsFlux", "FixedAPECUnabsFlux_pos", "FixedAPECUnabsFlux_neg",
                    "InterpPowECFO", "InterpPowECFU", "InterpPowNH", "InterpPowNH_pos",
                    "InterpPowNH_neg", "InterpPowGamma", "InterpPowGamma_pos",
                    "InterpPowGamma_neg", "InterpPowFlux", "InterpPowFlux_pos",
                    "InterpPowFlux_neg", "InterpPowUnabsFlux", "InterpPowUnabsFlux_pos",
                    "InterpPowUnabsFlux_neg", "InterpAPECECFO", "InterpAPECECFU",
                    "InterpAPECNH", "InterpAPECNH_pos", "InterpAPECNH_neg", "InterpAPECkT",
                    "InterpAPECkT_pos", "InterpAPECkT_neg", "InterpAPECFlux", "InterpAPECFlux_pos",
                    "InterpAPECFlux_neg", "InterpAPECUnabsFlux", "InterpAPECUnabsFlux_pos",
                    "InterpAPECUnabsFlux_neg", "P_pow", "P_APEC", "FittedPowECFO", "FittedPowECFU",
                    "FittedPowNH", "FittedPowNH_pos", "FittedPowNH_neg", "FittedPowGamma",
                    "FittedPowGamma_pos", "FittedPowGamma_neg", "FittedPowFlux",
                    "FittedPowFlux_pos","FittedPowFlux_neg", "FittedPowUnabsFlux",
                    "FittedPowUnabsFlux_pos", "FittedPowUnabsFlux_neg", "FittedPowCstat",
                    "FittedPowDOF", "FittedPowReducedChi2", "FittedAPECECFO", "FittedAPECECFU",
                    "FittedAPECNH", "FittedAPECNH_pos", "FittedAPECNH_neg", "FittedAPECkT",
                    "FittedAPECkT_pos", "FittedAPECkT_neg", "FittedAPECFlux", "FittedAPECFlux_pos",
                    "FittedAPECFlux_neg", "FittedAPECUnabsFlux", "FittedAPECUnabsFlux_pos",
                    "FittedAPECUnabsFlux_neg", "FittedAPECCstat", "FittedAPECDOF",
                    "FittedAPECReducedChi2"]

swift_data.remove_columns(columns_to_remove)

# set masks to median values
for col_name in swift_data.colnames:
    if not isinstance(swift_data[col_name], MaskedColumn):
        continue
    median = np.median(swift_data[col_name][~swift_data[col_name].mask])
    swift_data[col_name][swift_data[col_name].mask] = median

print("rows", len(swift_data))
print("cols", len(swift_data.colnames))
swift_data.write(os.path.dirname(os.getcwd()) + "\\Catalogues\\Swift\\swift_data.fits", format = "fits", overwrite = True)