
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#if [ $CUDA_VISIBLE_DEVICES != 0 ] ; then
#       # Exit with status 99, which tells the scheduler to resubmit the job
#       exit 99
#fi

nvidia-smi

singularity exec --nv  ../../../ngc_container/ python3 ../examples/main_training_testing_split_DeepConvSurv_autoEncoderSegSkipConnection_imaging_clinics_OS.py


