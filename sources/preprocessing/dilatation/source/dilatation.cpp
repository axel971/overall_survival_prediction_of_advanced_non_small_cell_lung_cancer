
#include <iostream>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkFlatStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"


using namespace std;
using namespace itk;


int main(int argc, char* argv[])
{

// Get the input mask
using pixelType = unsigned int;
using ImageType = itk::Image<pixelType, 3> ;
using ReaderTypeImage = itk::ImageFileReader<ImageType>;

ReaderTypeImage::Pointer readerImage = ReaderTypeImage::New();
readerImage->SetFileName(argv[1]);
readerImage->Update();


using StructuringElementType = FlatStructuringElement<3>;
StructuringElementType::RadiusType radius;
radius.Fill(5);
StructuringElementType  structuringElement = StructuringElementType::Ball(radius);

using BinaryDilateImageFilterType = itk::BinaryDilateImageFilter<ImageType, ImageType, StructuringElementType>;

BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
dilateFilter->SetInput(readerImage->GetOutput());
dilateFilter->SetKernel(structuringElement);
dilateFilter->SetForegroundValue(1);


//Save the region of interest
using WriterTypeImage = itk::ImageFileWriter<ImageType>;
WriterTypeImage::Pointer writer = WriterTypeImage::New();
writer->SetFileName(argv[2]);
writer->SetInput(dilateFilter->GetOutput());
writer->Update();

return 1;
}
