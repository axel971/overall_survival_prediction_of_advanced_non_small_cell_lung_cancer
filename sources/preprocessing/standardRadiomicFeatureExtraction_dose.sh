


singularity exec ../../../pyradiomics_container/ pyradiomics ../../data/raw_data/listNameFiles/imageAndMaskPathsForRadiomicFeatureComputation_dose.csv  -p ./radiomic_params.yaml -o ../../data/radiomic_features/radiomicFeatures_dose.csv -f csv --jobs 3

