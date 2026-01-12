# AGN-Predictor
Introduction:
The goal of this project is to use machine learning models trained off of telescope and satellite data to classify previously unknown sources in the sky. Currently the only classification is active galaxy nuclei or not.

This project is heavily inspired by Dr. Sabrina Einecke's 2017 dissertation "A Data Mining Guide to the Galaxy" where she utilizes multi-wavelength data to make AGN and blazar random forest classification models.

Classification:
Active galaxy nucleus (AGN), or opposingly, non active galaxy nucleus (NAG)

Model and essential libraries:
Random forest classifier from the scikit-learn python library.
astropy library to read .fits files into astropy Tables for processing
numpy and pandas libraries for more data processing

Data:
The data which the model trains and predicts on uses data from multiple wavelengths of light each acquired from a different astronomy catalogue. The two current catalogues used are the LAT 14-year Source Catalogue (4FGL-DR4) from the Fermi Gamma-ray Space Telescope and the Second Swift-XRT Point Source Catalog (2SXPS) from the Swift X-ray Telescope. These two catalogues contain source detections in the Gamma-ray and X-ray wavelengths respectively which are the two most significant emission wavelengths of active galaxies. More wavelengths can also be used such as infared and radio seen in Dr. Einecke's paper.

In order to utilize data from different catalogues, sources from different catalogues must be cross-matched. The Fermi catalogue, which has the least amount of sources and contains AGN, NAG, and unknown classifications already, is used as the main catalogue from which other catalogues are cross-matched to. Cross-matching in this context means matching each source in the Fermi catalogue to its corresponding source in the Swift catalogue which refers to the same object in the sky. Each Fermi source is defined by a 95% error ellipse, so the first step in cross-matching is getting every Swift source in every Fermi 95% error ellipse, and then choosing one Swift source from those candidates to be used as the true counterpart for each Fermi source. Before cross-matching, the catalogues need to first be filtered.

Filtering catalogues (CatalogueMatch.py):
The following are the features and thresholds used to trim the catalogues down. This is done so that we have higher quality data to train the model on.

Fermi:
Signif_Avg: we only keep Fermi sources that have values for the Signif_Avg feature. We remove all of the rows with masked values. We then only keep rows that have Signif_Avg > 5, corresponding to robust detections.
ROI_num: we only keep rows with ROI_num >= 0, as sources with negative values do not have reliable spectral or variability measurements and are not meaningful.
Extended_Source_Name: we then only keep sources that have an empty Extended_Source_Name because active galaxies are point sources and not extended sources.

Swift:
DetFlag: we keep sources with 0 as the DetFlag because these correspond to good detections with no known problems with the source.
FieldFlag: we keep sources with 0 as the FieldFlag because this correspdonds to no known problems with the observation of the source.
OpticalLoadingWarning: we keep 0 values for OpticalLoadWarning
StrayLightWarning: we keep 0 values for SrayLightWarning
NearBrightSourceWarning: we keep 0 values for NearBrightSourceWarning
IsPotentialAlias: we keep 0 values for IsPotentialAlias
NearestNeighbour and Err90: we keep sources such that the distance to the nearest neighbour is greater than the 90% error circle of the source, NearestNeighbour > Err90.
Exposure: we keep sources with an exposure of at least 500 so that we do not include sources with low exposure time and unreliable data, Exposure > 500.

These filter cuts are applied to both catalogues in their entirety which may be too harsh. A future improvement may be to do these cuts for the training data, but then not as many of these cuts for the application data that the model uses to predict source classifications.

With these cuts, the Fermi catalogue gets reduced from 7,195 sources to 5,616 sources and Swift gets reduced from 146,768 sources to 131,039 sources. Of the 5,616 Fermi sources, 3,472 of them are AGN, 554 of them are NAG, and 1,590 are UNK (unknown classification).

Cross-matching (CatalogueMatch.py):
Using the filtered catalogues, each Fermi source is then matched to Swift sources. This is simply done by getting every Swift source that resides in each Fermi 95% ellipse.

The algorithm used to do this efficiently, which is especially important for bigger catalogues with millions of sources, is using HEALPix from astropy_healpix to divide the sky into pixel regions. HEALPix can convert sources with right ascension and declination coordinates to a particular pixel region in the sky. Then, using Python's defaultdict(list) from the collections library, we can store a list of Swift sources that reside in each pixel region for immediate lookup (constant time complexity). That way, instead of looping through every Swift source when trying to match to a Fermi source, we can just loop through all of the Swift sources in the same pixel region as the Fermi source and match all of those Swift sources that reside in the Fermi 95% ellipse. This is done in the match_counterparts() function in CatalogueMatch.py. Only the indices of the Fermi and Swift sources are matched which can be used later to get the actual matched Fermi and Swift data rows.

Out of 5,616 Fermi sources and 131,039 Swift sources, 3,019 Swift sources are matched to 1,967 Fermi sources. Most Fermi sources do not get matched with any Swift sources, but for those that do, we average 1.535 Swift sources per Fermi source. The Fermi sources not matched with any Swift sources are not used for model training or application because they don't have the necessary X-ray features.

Catalogue merging (CatalogueCombine.py):
In this section, we used the cross-matching from the previous section to combine the catalogues together. To combine the catalogues together, we need to basically add the Swift features from the matched Swift source to each Fermi source. To do this, we need to first select the correct Swift source from the possible matched counterparts for each Fermi source.

Fermi's RA_Counterpart and DEC_Counterpart features are the coordinates of the most likely associated counterpart of the gamma-ray source. When these coordinates are available or when the probability of association is high enough, these associated positions can be used to determine the most likely Swift counterpart. Simply, whichever matched Swift counterpart in the error ellipse of the Fermi source is closest to this associated position is chosen as the true Swift counterpart. If the association probability is too low or these coordinates don't exist, then the Swift source with the highest likelihood of being the true counterpart is selected according to its position within the 95% error ellipse.

Once we have one matched Swift source for every Fermi source, we can combine the catalogues together. To do this efficiently, we can use the astropy hstack() function which appends one astropy table to another. We can append the tables column-wise once the two tables are matched by row (each row of the Swift table is the matched Swift source of the Fermi source of the same row). Simply put, each Fermi source then gains the columns from its matched Swift source.

Once the catalogues are combined, and after pre-processing which is explained in the next section, our combined Fermi-Swift table has 1,967 rows and 126 columns.

Pre-processing (FermiPreProcess.py and SwiftPreProcess.py):

Parameter Tuning (GridSearch.py):

Feature Selection (FeatureSelection.py):

Application (ModelApplication.py):
