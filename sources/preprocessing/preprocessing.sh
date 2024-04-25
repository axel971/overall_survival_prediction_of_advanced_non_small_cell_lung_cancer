

#path=`readlink -f "${BATCH_SOURCE:-$0}"`
#ROOT_DIR=`dirname $path`
ROOT_DIR=$PWD
ROOT_DIR=$ROOT_DIR"/.."


subject_list=$ROOT_DIR"/../data/raw_data/listNameFiles/patientIDs.csv"

for iSubject in `cat $subject_list`
do
	iSubject=`echo $iSubject | cut -d ',' -f2`
	
	if [[ $iSubject != 'subject_name' ]]
	then

        	echo $iSubject
        	qsub -l h_vmem=80G -o $ROOT_DIR/preprocessing/output_qsub_files/$iSubject".stdout" -e $ROOT_DIR/preprocessing/output_qsub_files/$iSubject"_err.stdout"   ./wrapper_preprocessing.sh $iSubject 
               #qsub -l h_vmem=80G -o $ROOT_DIR/preprocessing/output_qsub_files/$iSubject".stdout" -e $ROOT_DIR/preprocessing/output_qsub_files/$iSubject"_err.stdout"   ./wrapper_preprocessing_2.sh $iSubject
               # ./wrapper_preprocessing.sh $iSubject
	fi

done
