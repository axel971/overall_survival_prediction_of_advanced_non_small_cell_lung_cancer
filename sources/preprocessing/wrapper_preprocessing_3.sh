


iSubject=$1


#path=`readlink -f "${BATCH_SOURCE:-$0}"`
#ROOT_DIR=`dirname $path`
ROOT_DIR=$PWD
ROOT_DIR=$ROOT_DIR"/.."

template_image="000746628"

input_images=$ROOT_DIR"/../data/raw_data/images"
input_delineations=$ROOT_DIR"/../data/raw_data/delineations"

histogram_matching_images=$ROOT_DIR"/../data/preprocessing/images/histogram_matching"
histogram_matching_build=$ROOT_DIR"/preprocessing/histogram_matching/build"

resampled_images=$ROOT_DIR"/../data/preprocessing/images/resampling"
resampled_delineations=$ROOT_DIR"/../data/preprocessing/delineations/resampling"
resampling_build=$ROOT_DIR"/preprocessing/resampling/build"
resampling_delineation_build=$ROOT_DIR"/preprocessing/resampling_delineation/build"

dilatation_build=$ROOT_DIR"/preprocessing/dilatation/build"
dilatation_delineations=$ROOT_DIR"/../data/preprocessing/delineations/dilatation"

extractConnectedComponent_build=$ROOT_DIR"/preprocessing/extractConnectedComponent/build"
extractConnectedComponent_delineations=$ROOT_DIR"/../data/preprocessing/delineations/extractConnectedComponent"

boundingBox_images=$ROOT_DIR"/../data/preprocessing/images/boundingBox"
boundingBox_delineations=$ROOT_DIR"/../data/preprocessing/delineations/boundingBox"
boundingBox_build=$ROOT_DIR"/preprocessing/boundingBox/build"

normalize_build=$ROOT_DIR"/preprocessing/normalize/build"
normalize_images=$ROOT_DIR"/../data/preprocessing/images/normalize"

cd $histogram_matching_build
   ./histogramMatching $input_images/$iSubject".nii.gz" $input_images/$template_image".nii.gz" $histogram_matching_images/$iSubject"_histogram_matching.nii.gz" 

cd $dilatation_build
 
 ./dilatation $input_delineations/$iSubject/"GTV.mha" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz"

cd $extractConnectedComponent_build
 ./extractConnectedComponent $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz" $extractConnectedComponent_delineations/$iSubject"_connectedComponent_delineation.nii.gz" 

cd $boundingBox_build
 ./boundingBox $histogram_matching_images/$iSubject"_histogram_matching.nii.gz" $extractConnectedComponent_delineations/$iSubject"_connectedComponent_delineation.nii.gz" $boundingBox_images/$iSubject"_boundingBox.nii.gz" 
 
 ./boundingBox $input_delineations/$iSubject/"GTV.mha" $extractConnectedComponent_delineations/$iSubject"_connectedComponent_delineation.nii.gz" $boundingBox_delineations/$iSubject"_boundingBox_delineation.nii.gz" 
  

cd $resampling_build
  ./resampling $boundingBox_images/$iSubject"_boundingBox.nii.gz" $resampled_images/$iSubject"_resampled_boundingBox.nii.gz" "128" "128" "64"
	
cd $resampling_delineation_build
  ./resampling_delineation $boundingBox_delineations/$iSubject"_boundingBox_delineation.nii.gz"	$resampled_delineations/$iSubject"_resampled_boundingBox_delineation.nii.gz" "128" "128" "64"

cd $normalize_build
 ./normalize $resampled_images/$iSubject"_resampled_boundingBox.nii.gz" $normalize_images/$iSubject"_resampled_normalized_boundingBox.nii.gz"
	
