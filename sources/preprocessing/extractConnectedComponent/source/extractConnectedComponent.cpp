
#include <iostream>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkImageMaskSpatialObject.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkLabelImageToShapeLabelMapFilter.h"

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


//Extract the largest connected component inside the mask image
using ConnectedComponentImageFilterType = itk::ConnectedComponentImageFilter<ImageType, ImageType>;
ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New();
connected->SetInput(readerImage->GetOutput());
connected->SetBackgroundValue(0);
connected->Update();


using LabelShapeKeepNObjectsImageFilterType = itk::LabelShapeKeepNObjectsImageFilter<ImageType>;
LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
labelShapeKeepNObjectsImageFilter->SetInput(connected->GetOutput());
labelShapeKeepNObjectsImageFilter->SetBackgroundValue(0);
labelShapeKeepNObjectsImageFilter->SetNumberOfObjects(connected->GetObjectCount());
labelShapeKeepNObjectsImageFilter->SetAttribute(LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
labelShapeKeepNObjectsImageFilter->Update();

using ShapeLabelObjectType = itk::ShapeLabelObject< pixelType, 3 >;
using LabelMapType = itk::LabelMap<ShapeLabelObjectType>;
using LabelImage2ShapeLabelMapFilterType = itk::LabelImageToShapeLabelMapFilter< ImageType, LabelMapType>;

LabelImage2ShapeLabelMapFilterType::Pointer labelImage2ShapeLabelMapFilter = LabelImage2ShapeLabelMapFilterType::New();
labelImage2ShapeLabelMapFilter->SetInput(labelShapeKeepNObjectsImageFilter->GetOutput());
LabelMapType::Pointer labelMap = labelImage2ShapeLabelMapFilter->GetOutput();
labelImage2ShapeLabelMapFilter->Update();

unsigned int nObjects2Delete = 0;
unsigned int minSize = 15;

for(unsigned int i = 0; i < labelMap->GetNumberOfLabelObjects(); i++)
{
     ShapeLabelObjectType* labelObject = labelMap->GetNthLabelObject(i);
     cout << "Number of voxel for object " << i << " : " << labelObject->GetNumberOfPixels() << endl;
     if(labelObject->GetNumberOfPixels() <  minSize)
      {
        nObjects2Delete++;
      }
}

cout << "Number of objects: " << labelShapeKeepNObjectsImageFilter->GetNumberOfObjects() << endl;
cout << "Number of object to delete: " << nObjects2Delete << endl;

labelShapeKeepNObjectsImageFilter->SetNumberOfObjects(connected->GetObjectCount() - nObjects2Delete);
labelShapeKeepNObjectsImageFilter->Update();

cout << "Number of objects to save: " << labelShapeKeepNObjectsImageFilter->GetNumberOfObjects() << endl;

using RescaleFilterType = itk::RescaleIntensityImageFilter< ImageType, ImageType>;
RescaleFilterType::Pointer rescaleFilter = RescaleFilterType ::New();
rescaleFilter->SetOutputMinimum(0);
rescaleFilter->SetOutputMaximum(itk::NumericTraits<pixelType>::max());
rescaleFilter->SetInput(labelShapeKeepNObjectsImageFilter->GetOutput());

//Save the region of interest
using WriterTypeImage = itk::ImageFileWriter<ImageType>;
WriterTypeImage::Pointer writer = WriterTypeImage::New();
writer->SetFileName(argv[2]);
writer->SetInput(rescaleFilter->GetOutput());
writer->Update();

return 1;
}
