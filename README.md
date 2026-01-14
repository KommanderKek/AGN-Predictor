# AGN-Predictor
Introduction:

The goal of this project is to use machine learning models trained off of telescope and satellite data to classify previously unknown sources in the sky. Currently the only classification is active galaxy nuclei or not.

This project is heavily inspired by Dr. Sabrina Einecke's 2017 dissertation "A Data Mining Guide to the Galaxy" where she utilizes multi-wavelength data to make AGN and blazar random forest classification models.

Classification:

Active galaxy nucleus (AGN) or non active galaxy nucleus (NAG). Some sources are classified as unknown (UNK), and the model, once trained and optimized, will predict these sources as AGN or NAG.

Model and essential libraries:

Random forest classifier from the scikit-learn python library.
astropy library to read .fits files into astropy Tables for processing
numpy and pandas libraries for more data processing

Data:

The data which the model trains and predicts on uses data from multiple wavelengths of light each acquired from a different astronomy catalogue. The two current catalogues used are the LAT 14-year Source Catalogue (4FGL-DR4) from the Fermi Gamma-ray Space Telescope and the Second Swift-XRT Point Source Catalog (2SXPS) from the Swift X-ray Telescope. These two catalogues contain source detections in the Gamma-ray and X-ray wavelengths respectively which are the two most significant emission wavelengths of active galaxies. More wavelengths can also be used such as infared and radio seen in Dr. Einecke's paper.

In order to utilize data from different catalogues, sources from different catalogues must be cross-matched. The Fermi catalogue, which has the least amount of sources and contains AGN, NAG, and UNK classifications already, is used as the main catalogue from which other catalogues are cross-matched to. Cross-matching in this context means matching each source in the Fermi catalogue to its corresponding source in the Swift catalogue which refers to the same object in the sky. Each Fermi source is defined by a 95% error ellipse, so the first step in cross-matching is getting every Swift source in every Fermi 95% error ellipse, and then choosing one Swift source from those candidates to be used as the true counterpart for each Fermi source. Before cross-matching, the catalogues need to first be filtered.

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

Once the catalogues are combined, and after pre-processing which is explained in the next section, our combined Fermi-Swift table has 1,967 rows and 126 columns. The table is made up of 1,630 AGN, 164 NAG, and 173 UNK.

Pre-processing (FermiPreProcess.py and SwiftPreProcess.py):

The purpose of filtering the catalogues was to remove low quality sources to train and predict over. This actively removed rows from the catalogues. This pre-processing section then focuses on performing finishing touches by handling incompatible values and removing uncessesary columns. This step is performed after filtering and matching because certain columns are needed for filtering and matching but need to be removed later for training and predicting. The catalogue merging step, which comes after pre-processing, also removes some more columns such as positional coordinate.

A lot of columns are removed from the Fermi and Swift catalogues, so refer to the pre-processing python files to see what columns were removed. Columns were removed that have no physical significance to predicting a source as AGN or not or that directly induce biases into the data. Examples of columns that were removed are identification and bookkeeping columns, positions and coordinates (except for error ellipses), associations and counterpart information, quality flags and warning indicators, and class-dependent columns. Fermi and Swift originally have a total of 79 and 230 columns respectively. After pre-processing and removing columns during the merging phase, Fermi and Swift have 42 and 84 columns respectively totaling 126 columns including the AGN/NAG/UNK classification column from the Fermi catalogue.

The other part of pre-processing is handling invalid values, including masked values, infinities, and non-numbers like strings. Machine learning models only work with real numbers and nothing else. There were no strings in any of the remaning column features, but to handle masked values and infinities, I simply set all of these values to the median value of their respective column. This means that, to the model, a sample with this median value for a particular feature implies that it's just a typical detection, and we avoid injecting artificial trends in the data. The median is better than the mean value because the mean value is greatly influenced by extreme values in the column, and the median value is a better statistic for the true central population of the column data.

Parameter Tuning (GridSearch.py):

Parameter tuning means to just tune the hyperparameters of the random forest classifier model to get the best possible performance. I use ROC AUC to measure model performance as this is the best way to measure the models ability to distinguish between classes accross all classification thresholds. To test model performance, I do a 5-fold cross validation over the AGN/NAG training set (using the combined Fermi-Swift data we created in the previous sections). The main hyperperameters I tune are num_estimators (number of trees), max_depth (maximum depth for each tree), and min_samples_split (minimum number of samples required to split a node in the trees). To find the best combination of hyperparameters, I simply do a parameter grid search, meaning that I loop through all possible hyperparameter combinations, and perform a 5-fold cross validation for each one and record the ROC AUC score. Whichever combination has the highest score is the set of hyperparameters that should be used for further analysis and model application. Note that the class_weight for the model should also be set to "balanced" or "balanced_subsample" because the class frequencies in the data is not balanced (much more AGN than NAG) which could lead to model biases. I do a 5-fold cross validation as opposed to, say, a 10-fold cross validation because the training set is relatively small with only 2,000 rows. Splitting into 5 folds means that each test set is 400 samples, and I think this is a good number of samples to better approximate the model performance. Any higher number of folds would reduce this test set size.

Using the right hyperparameters, I get an average ROC AUC of 0.95+ using cross validation on the training set.

This is a great result, and at this point the model tuning is basically done. However, there is more we can do in the next section to reduce the number of columns to simplify the data further. I tune the hyperparameters twice. I first tune using the combined Fermi-Swift data to get a good initial set of hyperparameters. I then use these hyperparameters in the next section during feature selection, and then I run a new grid search on the final data set with reduced features to get the hyperparameters of the final model. The hyperparameters found before feature selection were 400 trees, minimum samples split of 2, and max depth of 14. After feature selection, the optimal hyperparameters were 600 trees, minimum samples split of 4, and no max depth (tree depth has no limit).

Feature Selection (FeatureSelection.py):

This section is not completely necessary to perform, but if we ever want to perform physics analysis on the model, it helps to only keep the few most important columns actually needed for active galaxy prediction such that the model still performs well (0.95+ ROC AUC). The technique used in this section is called backwards elimination, where we remove a column, test the model performance using cross validation, record the performance, and then iteratively repeat until no more columns are left to remove. That way, we can see at what number of columns the model starts to significantly decrease in performance, and only keep those columns needed for performance.

We do not randomly remove columns though. Instead, after training the random forest classifier for each iteration, we remove the column of least importance according to the feature_importances_ attribute. We first prioritize columns that are highly correlated to other columns using Spearman's rank corellation coefficient because we want to get rid of redundant columns. In each backwards elimination iteration, we eliminate the column of lowest importance that is highely correlated to another column (rank coefficient > 0.95). Once there are no longer any highly correlated columns, we just remove the column of least importance.

I found that the there are only 3 - 5 or so columns that are extremely important for AGN classification and model performance which, in order of importance, are: PLEC_ExpfactorS, Frac_Variability, LP_beta, Unc_PLEC_EPeak, and Variability_Index all of which are features from the Fermi catalogue. Although only a small number of features contribute most strongly to the model performance, I retain a broader subset of the most important features (approximately 16) to reduce the risk of over-pruning and to allow for further analysis of the model results and features. These 16 features also includes some Swift features such as Exposure, Rate_band3_neg, BgCounts_band1, and BgCounts_band2.

While the classification is primarily driven by gamma-ray variability and spectral properties from the Fermi catalogue, a small number of Swift features were retained by the model. These features, such as exposure time, background counts, and high-energy rate uncertainties, do not directly encode source class, but instead provide information about the reliability and signal quality of the X-ray counterpart. Their inclusion improves classification robustness, particularly for sources with ambiguous gamma-ray properties.

Application (ModelApplication.py):

The following are the results. The final dataset used is the merged Fermi-Swift table made up of 1,630 AGN, 164 NAG, 173 UNK, and 16 feature columns. The model was trained on training subset of this dataset including all of the known classifications (1,630 AGN and 164 NAG), using n_estimators = 600, min_samples_split = 4, max_depth = None, and class_weight = "balanced_subsample." The model is then applied to the 173 UNK samples to predict whether they are AGN or NAG. For repeatability to my results, I used random_state = 1

Out of the 173 UNK sources, 108 of them are classified as high confidence AGN, 11 of them low confidence AGN, 37 high confidence NAG, and 17 low confidence NAG. High confidence meaning that the probability of that particular class >= 0.7, and low confidence meaning probability >= 0.5 but < 0.7.

Below are the specific ID Names, data release, and RA and DEC coordinates in degrees of the prediction results

high conf AGN, 108 sources:

4FGL J0057.9+6326 1 14.4955 63.4385

4FGL J0058.3-4603 1 14.5904 -46.056

4FGL J0059.4-5654 1 14.8643 -56.9089

4FGL J0117.9+1430 3 19.484 14.5001

4FGL J0132.1-0956 1 23.0306 -9.9386

4FGL J0132.9-4259 3 23.2374 -42.9834

4FGL J0133.0+5931 2 23.2607 59.5319

4FGL J0150.4+4848 1 27.6144 48.814

4FGL J0156.3-2420 1 29.0777 -24.3345

4FGL J0240.2-0248 1 40.0534 -2.8086

4FGL J0251.1-1830 1 42.7835 -18.5093

4FGL J0259.0+0552 1 44.7519 5.878

4FGL J0300.4+3450 2 45.1176 34.8442

4FGL J0359.7+4445 1 59.9473 44.7569

4FGL J0406.2+0639 1 61.5514 6.6524

4FGL J0544.4+2238 1 86.1093 22.6418

4FGL J0610.8-4911 1 92.7142 -49.1837

4FGL J0617.6-4028 1 94.4106 -40.4803

4FGL J0639.1-8009 3 99.7906 -80.1529

4FGL J0641.4+3349 3 100.3562 33.8195

4FGL J0647.6+4459 3 101.9061 44.985

4FGL J0650.6+2055 1 102.658 20.9266

4FGL J0653.4+1813 3 103.365 18.2265

4FGL J0659.9-2900 4 104.9828 -29.0015

4FGL J0704.3-4829 1 106.0841 -48.4846

4FGL J0725.7-0549 1 111.4493 -5.8193

4FGL J0737.4+6535 1 114.3737 65.5912

4FGL J0747.2-0736 1 116.8101 -7.6092

4FGL J0753.1-2624c 2 118.2895 -26.4008

4FGL J0800.9+0733 1 120.2262 7.5509

4FGL J0809.8+0507 1 122.4532 5.1313

4FGL J0810.0-0505 2 122.5079 -5.0987

4FGL J0839.6-8341 4 129.9157 -83.6924

4FGL J0852.5-2515 3 133.1457 -25.2523

4FGL J0903.5+4057 1 135.8994 40.9621

4FGL J0928.4-5256 1 142.1089 -52.9371

4FGL J0929.3-4110 2 142.3345 -41.1833

4FGL J0957.7-3510 2 149.426 -35.173

4FGL J0958.6-0522 2 149.6642 -5.3833

4FGL J1011.1-4420 1 152.7956 -44.3469

4FGL J1016.1-4247 1 154.0399 -42.7854

4FGL J1024.5-5329 1 156.1299 -53.496

4FGL J1024.5-4543 1 156.1479 -45.7311

4FGL J1036.5-7434c 4 159.1404 -74.5793

4FGL J1048.4-5030 1 162.1067 -50.5132

4FGL J1054.7-6008c 1 163.6789 -60.1401

4FGL J1058.4-6625 1 164.6112 -66.4293

4FGL J1106.4+0859 3 166.6213 8.9901

4FGL J1123.8-4552 1 170.958 -45.8714

4FGL J1131.6+4657 3 172.9014 46.958

4FGL J1146.0-0638 1 176.5016 -6.6381

4FGL J1155.2-1111 1 178.8195 -11.189

4FGL J1208.2+1158 1 182.0697 11.9734

4FGL J1220.1-2458 1 185.0473 -24.973

4FGL J1224.6+7011 1 186.1736 70.1965

4FGL J1250.9-4943 1 192.7265 -49.7203

4FGL J1256.9+2736 2 194.2417 27.6076

4FGL J1410.7+7405 1 212.6869 74.0899

4FGL J1419.2-6029 1 214.8085 -60.4891

4FGL J1438.0+0219 2 219.5039 2.3272

4FGL J1452.0-4148 1 223.0172 -41.8036

4FGL J1514.1-5805c 2 228.5442 -58.088

4FGL J1514.8+4448 1 228.7193 44.8105

4FGL J1555.3+2903 3 238.8319 29.0564

4FGL J1626.0-4917c 1 246.5083 -49.2838

4FGL J1626.5-4406 1 246.6427 -44.1031

4FGL J1647.5-5724 2 251.8835 -57.4067

4FGL J1648.1-1547 1 252.0274 -15.793

4FGL J1657.3-6232 2 254.3257 -62.5377

4FGL J1706.5-4023c 1 256.6436 -40.3847

4FGL J1719.1-5348 1 259.7793 -53.807

4FGL J1720.6-5144 1 260.1582 -51.7479

4FGL J1728.8-5131 2 262.2176 -51.5199

4FGL J1729.2-3641c 1 262.3107 -36.6879

4FGL J1729.9-4148 1 262.4798 -41.8042

4FGL J1740.7-6750 1 265.1841 -67.8423

4FGL J1743.8-3143 1 265.972 -31.7266

4FGL J1806.1-3344 1 271.5411 -33.7365

4FGL J1809.2-2726 1 272.3076 -27.4456

4FGL J1813.7-6846 1 273.4481 -68.7785

4FGL J1817.2-3035 1 274.3076 -30.5842

4FGL J1817.2-1946c 3 274.312 -19.78

4FGL J1821.6+6636 1 275.414 66.6093

4FGL J1856.1-1222 1 284.0329 -12.3751

4FGL J1910.8+2856 1 287.7156 28.9432

4FGL J1918.0+0331 1 289.5109 3.521

4FGL J1927.5+0154 1 291.876 1.9017

4FGL J1943.9+2841 1 295.9803 28.6957

4FGL J2016.3+4953 1 304.086 49.8931

4FGL J2027.0+3343 1 306.7703 33.7222

4FGL J2030.5+2235 1 307.6351 22.5945

4FGL J2037.0-2826 1 309.2524 -28.4383

4FGL J2105.9+7508 1 316.4812 75.1444

4FGL J2109.3+3531 4 317.3316 35.5242

4FGL J2109.6+3954 1 317.414 39.9103

4FGL J2122.3+7653 2 320.5892 76.8887

4FGL J2143.4+6608c 1 325.8546 66.1368

4FGL J2152.0+4718c 1 328.0106 47.3067

4FGL J2153.0-6442 3 328.2527 -64.7107

4FGL J2159.6-4620 1 329.9124 -46.3338

4FGL J2209.2+3710 1 332.3052 37.1791

4FGL J2217.5+6346 1 334.394 63.776

4FGL J2229.3+5348 3 337.3419 53.8151

4FGL J2303.9+5554 1 345.9964 55.9139

4FGL J2323.9+2519 1 350.9781 25.3322

4FGL J2336.9-8427 1 354.2359 -84.4618

4FGL J2337.7+1449c 3 354.4258 14.8192

4FGL J2347.9-5106 1 356.9985 -51.1139



low conf AGN, 11 sources:

4FGL J0838.4-3952 1 129.6142 -39.8762

4FGL J1312.3-6257 1 198.0953 -62.961

4FGL J1349.1-5829 1 207.283 -58.4893

4FGL J1717.6-4404 1 259.4006 -44.0778

4FGL J1749.8-0303 1 267.4622 -3.0518

4FGL J1809.8-2408c 2 272.4555 -24.1434

4FGL J1831.3-0203c 3 277.8287 -2.0565

4FGL J1845.3+5605 3 281.3458 56.0903

4FGL J1928.4+1801c 1 292.1115 18.0192

4FGL J1929.8+1832 1 292.4541 18.5499

4FGL J2210.2+5919 2 332.5548 59.3247



high conf NAG, 37 sources

4FGL J0002.1+6721c 1 0.5419 67.3578

4FGL J0608.8+2034c 1 92.2008 20.5771

4FGL J0658.7-0407c 4 104.6788 -4.1187

4FGL J0844.9-4117 1 131.2473 -41.2962

4FGL J0933.8-6232 1 143.475 -62.5499

4FGL J1008.1-5706c 1 152.0435 -57.1029

4FGL J1126.0-5007 1 171.5145 -50.1194

4FGL J1312.6-6231c 1 198.1557 -62.5281

4FGL J1321.1-6239 2 200.2867 -62.6592

4FGL J1357.3-6123 1 209.3319 -61.3919

4FGL J1416.7-5023 1 214.1781 -50.3895

4FGL J1424.2-6111c 3 216.0641 -61.1977

4FGL J1526.6-3810 1 231.6594 -38.169

4FGL J1529.4-6027 1 232.3669 -60.4583

4FGL J1611.9-5125c 1 242.9829 -51.4209

4FGL J1612.1+1407 1 243.0313 14.1168

4FGL J1620.8-4958c 1 245.2116 -49.9699

4FGL J1628.0-4920 1 247.0121 -49.3367

4FGL J1636.9-4710c 1 249.2441 -47.1687

4FGL J1639.8-4642c 1 249.9532 -46.7141

4FGL J1645.8-4533c 1 251.4549 -45.5536

4FGL J1701.9-4625 1 255.4959 -46.4318

4FGL J1714.9-3324 1 258.7381 -33.4135

4FGL J1729.9-2403 1 262.4984 -24.0511

4FGL J1739.5-2929 1 264.8775 -29.4858

4FGL J1741.6-2730 1 265.4124 -27.5098

4FGL J1808.4-3358 1 272.1018 -33.9817

4FGL J1826.5-1202c 1 276.6486 -12.0484

4FGL J1842.5-0359c 1 280.6372 -3.9882

4FGL J1844.4-0306 1 281.1188 -3.1161

4FGL J1848.6-0202c 1 282.166 -2.0493

4FGL J1904.7-0708 1 286.1935 -7.1373

4FGL J2052.3+4437 1 313.082 44.6333

4FGL J2054.2+6904 1 313.5671 69.0769

4FGL J2056.4+4351c 1 314.1141 43.855

4FGL J2212.4+0708 1 333.1083 7.1428

4FGL J2250.5+3305 1 342.6404 33.0989



low conf NAG, 17 sources

4FGL J0031.0-2327 1 7.7564 -23.4581

4FGL J0608.6-2305 2 92.1506 -23.0918

4FGL J0859.2-4729 1 134.8042 -47.4883

4FGL J1325.4-4706 4 201.3618 -47.1146

4FGL J1350.9-2757 1 207.7332 -27.9534

4FGL J1407.7-3017 1 211.9374 -30.2858

4FGL J1700.2-4237c 1 255.0626 -42.6282

4FGL J1703.6-2850 1 255.9146 -28.8346

4FGL J1742.5-2833 1 265.6335 -28.5588

4FGL J1752.3-2914 1 268.0772 -29.2492

4FGL J1757.6-2731 1 269.4012 -27.5282

4FGL J1817.7-2517 2 274.4431 -25.288

4FGL J1846.9-0227 1 281.7299 -2.4522

4FGL J1855.2+0456 1 283.8159 4.9427

4FGL J2059.7+4837 2 314.9363 48.6233

4FGL J2241.4-8327 1 340.3568 -83.4612

4FGL J2315.9+5955c 1 348.9794 59.9231
