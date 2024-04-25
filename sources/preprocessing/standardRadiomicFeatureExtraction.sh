


singularity exec ../../../pyradiomics_container/ pyradiomics ../../data/raw_data/listNameFiles/imageAndMaskPathsForRadiomicFeatureComputation.csv  -p ./radiomic_params.yaml -o ../../data/radiomic_features/radiomicFeatures.csv -f csv --jobs 3

