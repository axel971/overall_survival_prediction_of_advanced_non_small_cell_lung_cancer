#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include <iostream>
#include <cstdlib>

using namespace std;
using namespace itk;

int main(int argc, char * argv[])
{

 typedef float voxelType; 

 typedef Image<voxelType, 3> imageType;
 typedef ImageFileReader<imageType> imageReaderType; 
 
 typedef IdentityTransform<double, 3> transformType;
 
 typedef LinearInterpolateImageFunction<imageType, double> interpolatorType;
 
 typedef ResampleImageFilter<imageType, imageType> resamplerType;

 // Load the input image
 imageReaderType::Pointer reader = imageReaderType::New();
 reader->SetFileName(argv[1]);
 reader->Update();
 
 // Instantiate the transformation
 transformType::Pointer transform = transformType::New();
 transform->SetIdentity();
 
 // Instantiate the B-Spline interpolator
 interpolatorType::Pointer interpolator = interpolatorType::New();

 
 // Get target voxel_size (spacing)
 double new_spacing[3];
 new_spacing[0] = atof(argv[3]);
 new_spacing[1] = atof(argv[4]);
 new_spacing[2] = atof(argv[5]);

 // Instantiate the resampler
 resamplerType::Pointer resampler = resamplerType::New();
 resampler->SetTransform(transform);
 resampler->SetInterpolator(interpolator);
 resampler->SetOutputOrigin(reader->GetOutput()->GetOrigin());
 
 Size<3>  new_size;
 new_size[0] = (unsigned int) (reader->GetOutput()->GetSpacing()[0] * (double)reader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]/ new_spacing[0]) ; 
 new_size[1] = (unsigned int) (reader->GetOutput()->GetSpacing()[1] * (double)reader->GetOutput()->GetLargestPossibleRegion().GetSize()[1]/ new_spacing[1]) ; 
 new_size[2] = (unsigned int) (reader->GetOutput()->GetSpacing()[2] * (double)reader->GetOutput()->GetLargestPossibleRegion().GetSize()[2]/ new_spacing[2]) ; 
 resampler->SetSize(new_size);
 resampler->SetOutputSpacing(new_spacing);
 resampler->SetInput(reader->GetOutput());
 resampler->SetOutputDirection(reader->GetOutput()->GetDirection());
 resampler->Update();


 // Write the resampled image
 typedef ImageFileWriter<imageType> writerType;
 writerType::Pointer writer = writerType::New();
 writer->SetFileName(argv[2]);
 writer->SetInput(resampler->GetOutput());
 writer->Update();
 
 return 0;

}
