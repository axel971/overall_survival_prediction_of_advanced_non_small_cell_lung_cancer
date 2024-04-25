


iSubject=$1


#path=`readlink -f "${BATCH_SOURCE:-$0}"`
#ROOT_DIR=`dirname $path`
ROOT_DIR=$PWD
ROOT_DIR=$ROOT_DIR"/.."
echo $ROOT_DIR

input_images=$ROOT_DIR"/../data/raw_data/images"
input_delineations=$ROOT_DIR"/../data/raw_data/delineations"
input_doses=$ROOT_DIR"/../data/raw_data/resampled_doses"

GTV_cleaning_delineations=$ROOT_DIR"/../data/preprocessing/delineations/clean_delineation"
GTV_cleaning_build=$ROOT_DIR"/preprocessing/GTVcleaning/build"

resampled_voxel_images=$ROOT_DIR"/../data/preprocessing/images/resampling_voxel"
resampled_voxel_delineations=$ROOT_DIR"/../data/preprocessing/delineations/resampling_voxel"
resampled_voxel_doses=$ROOT_DIR"/../data/preprocessing/doses/resampling_voxel"
resampling_voxel_build=$ROOT_DIR"/preprocessing/resampling_voxel/build"
resampling_voxel_delineation_build=$ROOT_DIR"/preprocessing/resampling_delineation_voxel/build"

dilatation_build=$ROOT_DIR"/preprocessing/dilatation/build"
dilatation_delineations=$ROOT_DIR"/../data/preprocessing/delineations/dilatation"

extractConnectedComponent_build=$ROOT_DIR"/preprocessing/extractConnectedComponent/build"
extractConnectedComponent_delineations=$ROOT_DIR"/../data/preprocessing/delineations/extractConnectedComponent"

boundingBox_images=$ROOT_DIR"/../data/preprocessing/images/boundingBox"
boundingBox_delineations=$ROOT_DIR"/../data/preprocessing/delineations/boundingBox"
boundingBox_build=$ROOT_DIR"/preprocessing/boundingBox/build"

maxBoundingBox_images=$ROOT_DIR"/../data/preprocessing/images/maxBoundingBox"
maxBoundingBox_delineations=$ROOT_DIR"/../data/preprocessing/delineations/maxBoundingBox"
maxBoundingBox_doses=$ROOT_DIR"/../data/preprocessing/doses/maxBoundingBox"
maxBoundingBox_build=$ROOT_DIR"/preprocessing/maxBoundingBox/build"

resampled_images=$ROOT_DIR"/../data/preprocessing/images/resampling"
resampled_delineations=$ROOT_DIR"/../data/preprocessing/delineations/resampling"
resampled_doses=$ROOT_DIR"/../data/preprocessing/doses/resampling"
resampling_build=$ROOT_DIR"/preprocessing/resampling/build"
resampling_delineation_build=$ROOT_DIR"/preprocessing/resampling_delineation/build"

normalize_build=$ROOT_DIR"/preprocessing/normalize/build"
normalize_images=$ROOT_DIR"/../data/preprocessing/images/normalize"

maskBoundingBox_build=$ROOT_DIR"/preprocessing/maskBoundingBox/build"
maskBoundingBox_images=$ROOT_DIR"/../data/preprocessing/images/maskedBoundingBox"

cd $GTV_cleaning_build
  ./GTVcleaning $input_delineations/$iSubject/"GTV.mha" $input_delineations/$iSubject/"PTV.mha" $GTV_cleaning_delineations/$iSubject"_delineation.nii.gz"

cd $resampling_voxel_build
  ./resampling_voxel $input_images/$iSubject".nii.gz" $resampled_voxel_images/$iSubject"_resampled.nii.gz" "1.102" "1.102" "2.960"
  ./resampling_voxel $input_doses/$iSubject"_dose.nii.gz" $resampled_voxel_doses/$iSubject"_resampled_doses.nii.gz" "1.102" "1.102" "2.960"
	
cd $resampling_voxel_delineation_build
  ./resampling_delineation_voxel $GTV_cleaning_delineations/$iSubject"_delineation.nii.gz" $resampled_voxel_delineations/$iSubject"_resampled_delineation.nii.gz" "1.102" "1.102" "2.960"


cd $dilatation_build 
  ./dilatation $resampled_voxel_delineations/$iSubject"_resampled_delineation.nii.gz" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz"

cd $boundingBox_build
   ./boundingBox $resampled_voxel_images/$iSubject"_resampled.nii.gz" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz" $boundingBox_images/$iSubject"_boundingBox.nii.gz" 

  ./boundingBox $resampled_voxel_delineations/$iSubject"_resampled_delineation.nii.gz" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz" $boundingBox_delineations/$iSubject"_boundingBox_delineation.nii.gz"

cd $maxBoundingBox_build
    ./maxBoundingBox $resampled_voxel_images/$iSubject"_resampled.nii.gz" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz" $maxBoundingBox_images/$iSubject"_maxBoundingBox.nii.gz" "332" "264" "92" "-1000"
    ./maxBoundingBox $resampled_voxel_doses/$iSubject"_resampled_doses.nii.gz" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz" $maxBoundingBox_doses/$iSubject"_maxBoundingBox_doses.nii.gz" "332" "264" "92" "0"             
   ./maxBoundingBox $resampled_voxel_delineations/$iSubject"_resampled_delineation.nii.gz" $dilatation_delineations/$iSubject"_dilated_delineation.nii.gz" $maxBoundingBox_delineations/$iSubject"_maxBoundingBox_delineation.nii.gz" "332" "264" "92" "0"

cd $resampling_build
   ./resampling $maxBoundingBox_images/$iSubject"_maxBoundingBox.nii.gz" $resampled_images/$iSubject"_resampled_boundingBox.nii.gz" "128" "128" "64"
   ./resampling $maxBoundingBox_doses/$iSubject"_maxBoundingBox_doses.nii.gz" $resampled_doses/$iSubject"_resampled_boundingBox_doses.nii.gz" "128" "128" "64"

cd $resampling_delineation_build
  ./resampling_delineation $maxBoundingBox_delineations/$iSubject"_maxBoundingBox_delineation.nii.gz" $resampled_delineations/$iSubject"_resampled_boundingBox_delineation.nii.gz" "128" "128" "64"

cd $normalize_build
 #./normalize $resampled_images/$iSubject"_resampled_boundingBox.nii.gz" $normalize_images/$iSubject"_resampled_normalized_boundingBox.nii.gz"
	
cd $maskBoundingBox_build
  ./maskBoundingBox $resampled_images/$iSubject"_resampled_boundingBox.nii.gz" $resampled_delineations/$iSubject"_resampled_boundingBox_delineation.nii.gz" $maskBoundingBox_images/$iSubject"_masked_resampled_boundingBox.nii.gz"

