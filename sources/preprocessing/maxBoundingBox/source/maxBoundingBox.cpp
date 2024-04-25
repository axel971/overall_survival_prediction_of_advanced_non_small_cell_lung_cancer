
#include <iostream>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkImageMaskSpatialObject.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConstantPadImageFilter.h"

using namespace std;
using namespace itk;


int main(int argc, char* argv[])
{

// Get the input image
using pixelType = float;
using ImageType = itk::Image<pixelType, 3> ;
using ReaderTypeImage = itk::ImageFileReader<ImageType>;

ReaderTypeImage::Pointer readerImage = ReaderTypeImage::New();
readerImage->SetFileName(argv[1]);
readerImage->Update();

// Get the input mask
using ImageMaskSpatialObject = itk::ImageMaskSpatialObject<3>;
using MaskType = ImageMaskSpatialObject::ImageType;
using ReaderTypeMask = itk::ImageFileReader<MaskType>;


ReaderTypeMask::Pointer readerMask = ReaderTypeMask::New();
readerMask->SetFileName(argv[2]);
readerMask->Update();

// Get the max bounding box size
unsigned int maxBoundingBoxSize[3];
maxBoundingBoxSize[0] = atoi(argv[4]);
maxBoundingBoxSize[1] = atoi(argv[5]);
maxBoundingBoxSize[2] = atoi(argv[6]);


// Build the bounding box
ImageMaskSpatialObject::Pointer imageMask = ImageMaskSpatialObject::New();
imageMask->SetImage(readerMask->GetOutput());
imageMask->Update();

using RegionType = itk::ImageRegion<3>;
RegionType boundingBox;
boundingBox = imageMask->GetAxisAlignedBoundingBoxRegion();

int xDelta, yDelta, zDelta;
xDelta = (int) ((maxBoundingBoxSize[0] - boundingBox.GetSize(0)) / 2);
yDelta = (int) ((maxBoundingBoxSize[1] - boundingBox.GetSize(1)) / 2);
zDelta = (int) ((maxBoundingBoxSize[2] - boundingBox.GetSize(2)) / 2);

boundingBox.SetIndex(0, boundingBox.GetIndex(0)  - xDelta);
boundingBox.SetIndex(1, boundingBox.GetIndex(1)  - yDelta);
boundingBox.SetIndex(2, boundingBox.GetIndex(2)  - zDelta);

boundingBox.SetSize(0, maxBoundingBoxSize[0]);
boundingBox.SetSize(1, maxBoundingBoxSize[1]);
boundingBox.SetSize(2, maxBoundingBoxSize[2]);

cout << boundingBox << endl;

//Padd input image
ImageType::SizeType lowerExtendRegion;
lowerExtendRegion[0] = maxBoundingBoxSize[0];
lowerExtendRegion[1] = maxBoundingBoxSize[1];
lowerExtendRegion[2] = maxBoundingBoxSize[2];
                                           
ImageType::SizeType upperExtendRegion;
upperExtendRegion[0] = maxBoundingBoxSize[0];
upperExtendRegion[1] = maxBoundingBoxSize[1];
upperExtendRegion[2] = maxBoundingBoxSize[2];
 
using PaddingFilterType = itk::ConstantPadImageFilter<ImageType, ImageType>;
auto paddingFilter = PaddingFilterType::New();
paddingFilter->SetInput(readerImage->GetOutput());
paddingFilter->SetPadLowerBound(lowerExtendRegion);
paddingFilter->SetPadUpperBound(upperExtendRegion);
paddingFilter->SetConstant(atof(argv[7]));
paddingFilter->Update();

cout << readerImage->GetOutput()->GetLargestPossibleRegion().GetSize() <<  endl;
cout << paddingFilter->GetOutput()->GetLargestPossibleRegion().GetSize() <<  endl;

//Extract the bounding box in the padded input image
using FilterType = itk::RegionOfInterestImageFilter<ImageType, ImageType>;
FilterType::Pointer filter = FilterType::New();
filter->SetInput(paddingFilter->GetOutput());
//filter->SetRegionOfInterest(boundingBox->ComputeMyBoundingBoxInIndexSpace());
filter->SetRegionOfInterest(boundingBox);

//Save the region of interest
using WriterTypeImage = itk::ImageFileWriter<ImageType>;
WriterTypeImage::Pointer writer = WriterTypeImage::New();
writer->SetFileName(argv[3]);
writer->SetInput(filter->GetOutput());
writer->Update();

return 1;
}
