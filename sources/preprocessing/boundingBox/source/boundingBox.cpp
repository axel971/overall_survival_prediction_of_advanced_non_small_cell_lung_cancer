
#include <iostream>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkImageMaskSpatialObject.h"
#include "itkRegionOfInterestImageFilter.h"


using namespace std;
using namespace itk;


int main(int argc, char* argv[])
{

// Get the input image
using pixelType = float;
using ImageType = itk::Image<pixelType, 3> ;
using ReaderTypeImage = itk::ImageFileReader<ImageType>;

ReaderTypeImage::Pointer readerImage1 = ReaderTypeImage::New();
readerImage1->SetFileName(argv[1]);
readerImage1->Update();

// Get the input mask
using ImageMaskSpatialObject = itk::ImageMaskSpatialObject<3>;
using MaskType = ImageMaskSpatialObject::ImageType;
using ReaderTypeMask = itk::ImageFileReader<MaskType>;


ReaderTypeMask::Pointer readerMask = ReaderTypeMask::New();

readerMask->SetFileName(argv[2]);
readerMask->Update();


// Build the bounding box
ImageMaskSpatialObject::Pointer boundingBox = ImageMaskSpatialObject::New();
boundingBox->SetImage(readerMask->GetOutput());
boundingBox->Update();

//Get the region of interest from the bounding box
using FilterType = itk::RegionOfInterestImageFilter<ImageType, ImageType>;
FilterType::Pointer filter = FilterType::New();
filter->SetInput(readerImage1->GetOutput());
//filter->SetRegionOfInterest(boundingBox->ComputeMyBoundingBoxInIndexSpace());
filter->SetRegionOfInterest(boundingBox->GetAxisAlignedBoundingBoxRegion());

//Save the region of interest
using WriterTypeImage = itk::ImageFileWriter<ImageType>;
WriterTypeImage::Pointer writer = WriterTypeImage::New();
writer->SetFileName(argv[3]);
writer->SetInput(filter->GetOutput());
writer->Update();

return 1;
}
