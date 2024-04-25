


iSubject=$1


#path=`readlink -f "${BATCH_SOURCE:-$0}"`
#ROOT_DIR=`dirname $path`
ROOT_DIR=$PWD
ROOT_DIR=$ROOT_DIR"/.."


input_images=$ROOT_DIR"/../data/raw_data/images"
input_delineations=$ROOT_DIR"/../data/raw_data/delineations"

#histogram_matching_images=$ROOT_DIR"/../data/preprocessing/images/histogram_matching"
#histogram_matching_build=$ROOT_DIR"/preprocessing/histogram_matching/build"

resampled_images=$ROOT_DIR"/../data/preprocessing_2/images/resampling"
resampled_delineations=$ROOT_DIR"/../data/preprocessing_2/delineations/resampling"
resampling_build=$ROOT_DIR"/preprocessing/resampling/build"
resampling_delineation_build=$ROOT_DIR"/preprocessing/resampling_delineation/build"

dilatation_build=$ROOT_DIR"/preprocessing/dilatation/build"
dilatation_delineations=$ROOT_DIR"/../data/preprocessing_2/delineations/dilatation"

extractConnectedComponent_build=$ROOT_DIR"/preprocessing/extractConnectedComponent/build"
extractConnectedComponent_delineations=$ROOT_DIR"/../data/preprocessing_2/delineations/extractConnectedComponent"

boundingBox_images=$ROOT_DIR"/../data/preprocessing_2/images/boundingBox"
boundingBox_delineations=$ROOT_DIR"/../data/preprocessing_2/delineations/boundingBox"
boundingBox_build=$ROOT_DIR"/preprocessing/boundingBox/build"

normalize_build=$ROOT_DIR"/preprocessing/normalize/build"
normalize_images=$ROOT_DIR"/../data/preprocessing/images/normalize"

cd $dilatation_build
  mkdir $dilatation_delineations 
 ./dilatation $input_images/$iSubject".nii.gz" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz"

cd $extractConnectedComponent_build
  mkdir $extractConnectedComponent_delineations
 ./extractConnectedComponent $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz" $extractConnectedComponent_delineations/$iSubject"_connectedComponent_delineation.nii.gz" 

cd $boundingBox_build
 mkdir $boundingBox_images
 ./boundingBox $input_images/$iSubject".nii.gz" $extractConnectedComponent_delineations/$iSubject"_connectedComponent_delineation.nii.gz" $boundingBox_images/$iSubject"_boundingBox.nii.gz" 
 
 mkdir $boundingBox_delineations
 ./boundingBox $input_delineations/$iSubject/"GTV.mha" $extractConnectedComponent_delineations/$iSubject"_connectedComponent_delineation.nii.gz" $boundingBox_delineations/$iSubject"_boundingBox_delineation.nii.gz" 
  

cd $resampling_build
  mkdir $resampled_images
  ./resampling $boundingBox_images/$iSubject"_boundingBox.nii.gz" $resampled_images/$iSubject"_resampled_boundingBox.nii.gz" "128" "128" "64"
	
cd $resampling_delineation_build
  mkdir $resampled_delineations
  ./resampling_delineation $boundingBox_delineations/$iSubject"_boundingBox_delineation.nii.gz"	$resampled_delineations/$iSubject"_resampled_boundingBox_delineation.nii.gz" "128" "128" "64"

cd $normalize_build
 #./normalize $resampled_images/$iSubject"_resampled_boundingBox.nii.gz" $normalize_images/$iSubject"_resampled_normalized_boundingBox.nii.gz"
	
