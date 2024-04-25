

path=`readlink -f "${BATCH_SOURCE:-$0}"`
ROOT_DIR=`dirname $path`
ROOT_DIR=$ROOT_DIR"/.."

subject_list=$ROOT_DIR"/../data/listNameFiles/patient_IDs.csv" 

input_images=$ROOT_DIR"/../data/raw_data/images"
input_delineations=$ROOT_DIR"/../data/raw_data/delineations"

data_2_dispatch_folder=$ROOT_DIR"/../data/raw_data/original_CTs_delineations"


for iSubject in `cat $subject_list`
do

    iSubject=`echo $iSubject | cut -d ',' -f2`
   
   if [[ $iSubject != 'subject_name' ]]
   then    
        mv  $data_2_dispatch_folder/$iSubject*/$iSubject".nii"* $input_delineations/$iSubject"_delineation.nii"
	
        mv  $data_2_dispatch_folder/$iSubject*/*".nii"* $input_images/$iSubject".nii"
   fi
done	
