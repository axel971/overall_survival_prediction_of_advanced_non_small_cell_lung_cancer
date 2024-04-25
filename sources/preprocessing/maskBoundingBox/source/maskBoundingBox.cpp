
#include <iostream>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkMultiplyImageFilter.h"

using namespace std;
using namespace itk;


int main(int argc, char* argv[])
{

// Get the input image
using pixelType = float;
using ImageType = itk::Image<pixelType, 3> ;
using ReaderTypeImage = itk::ImageFileReader<ImageType>;

// Get input image 1
ReaderTypeImage::Pointer readerImage1 = ReaderTypeImage::New();
readerImage1->SetFileName(argv[1]);
readerImage1->Update();

// Get input image 2
ReaderTypeImage::Pointer readerImage2 = ReaderTypeImage::New();
readerImage2->SetFileName(argv[2]);
readerImage2->Update();
readerImage2->GetOutput()->SetOrigin(readerImage1->GetOutput()->GetOrigin());

//Multiply image 1 and image 2
using FilterType = itk::MultiplyImageFilter<ImageType, ImageType, ImageType>;
auto filter = FilterType::New();
filter->SetInput1(readerImage1->GetOutput());
filter->SetInput2(readerImage2->GetOutput());

//Save the region of interest
using WriterTypeImage = itk::ImageFileWriter<ImageType>;
WriterTypeImage::Pointer writer = WriterTypeImage::New();
writer->SetFileName(argv[3]);
writer->SetInput(filter->GetOutput());
writer->Update();

return 1;
}
