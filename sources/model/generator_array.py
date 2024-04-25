
import numpy as np
from tensorflow.keras.utils import Sequence
from model.one_hot_label import multi_class_labels
from model.dataio import write_label_nii, write_nii
from model.patch3d import patch

import torchio as tio
import random

class Generator3D(Sequence):
   
    def __init__(self, X, times, events, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.70)
	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        
        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        #while ( np.abs(np.sum(self.events[batch_indices]) - (self.batch_size //2) ) > 1e-5):
        #   batch_indices = np.array(random.sample(self.indices, self.batch_size))
    
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
       
  
        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)))   
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            
        return [X, events], times
        
       
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
        

class Generator3D_imaging_dose(Sequence):
   
    def __init__(self, X, doses, times, events, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.doses = doses[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.70)

	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        doses = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)

        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
          
        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            subject_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)), dose = tio.ScalarImage(tensor = np.expand_dims(self.doses[image_index], 0)))   
            subject_tio_augmented = self.data_augmentation_transform(subject_tio)
            X[i_batch, :, :, :, 0] = subject_tio_augmented['image'].numpy()[0]
            doses[i_batch, :, :, :, 0] = subject_tio_augmented['dose'].numpy()[0]
            
        return [X, doses, events], times
        
       
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1


class Generator3D_imaging_dose_clinics(Sequence):
   
    def __init__(self, X, doses, clinics, times, events, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.doses = doses[sorted_indices]
        self.clinics = clinics[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.70)

	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        doses = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)

        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
        clinics = self.clinics[sorted_batch_indices]
  
        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            subject_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)), dose = tio.ScalarImage(tensor = np.expand_dims(self.doses[image_index], 0)))   
            subject_tio_augmented = self.data_augmentation_transform(subject_tio)
            X[i_batch, :, :, :, 0] = subject_tio_augmented['image'].numpy()[0]
            doses[i_batch, :, :, :, 0] = subject_tio_augmented['dose'].numpy()[0]
            
        return [X, doses, clinics, events], times
        
       
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1

class Generator3D_OS_oneClassificationTask(Sequence):
   
    def __init__(self, X, times, events, classificationTask, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.classificationTask = classificationTask[sorted_indices]

        self.indices = range(self.n_subject)

        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        
        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
        classificationTask = self.classificationTask[sorted_batch_indices]

        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            
        return [X, events, classificationTask], times
        
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1


class Generator3D_seg(Sequence):
   
    def __init__(self, X, Y, times, events, labels=[1], batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.labels = labels

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.Y = Y[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.65)

	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], len(self.labels)), dtype= np.float32)

        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
       
  
        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)), segmentation = tio.LabelMap(tensor = np.expand_dims(self.Y[image_index], 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            Y[i_batch] = multi_class_labels(patch_tio_augmented['segmentation'].numpy()[0], self.labels)  
            
        return [X, events, Y], times
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
        


class Generator3D_seg_imaging_clinics(Sequence):
   
    def __init__(self, X, Y, clinics, times, events, labels=[1], batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.labels = labels

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.Y = Y[sorted_indices]
        self.clinics = clinics[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.65)

	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], len(self.labels)), dtype= np.float32)

        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
        clinics = self.clinics[sorted_batch_indices]
  
        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)), segmentation = tio.LabelMap(tensor = np.expand_dims(self.Y[image_index], 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            Y[i_batch] = multi_class_labels(patch_tio_augmented['segmentation'].numpy()[0], self.labels)  
            
        return [X, events, Y, clinics], times
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
        



class Generator3D_imaging_clinics(Sequence):
   
    def __init__(self, X, clinics, times, events, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.clinics = clinics[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.65)

	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype = np.float32)
        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
        clinics = self.clinics[sorted_batch_indices]
       # print(clinics)  

        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            
        return [X, clinics, events], times
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
        


class Generator3D_oneClassificationTask_imaging_clinics(Sequence):
   
    def __init__(self, X, clinics, times, events, classificationTask, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X = X[sorted_indices]
        self.clinics = clinics[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.classificationTask = classificationTask[sorted_indices]

        self.indices = range(self.n_subject)

        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        
        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
        clinics = self.clinics[sorted_batch_indices]
        classificationTask = self.classificationTask[sorted_batch_indices]

        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            
        return [X, clinics, events, classificationTask], times
        
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1


class Generator3D_oneClassificationTask_2imaging_clinics(Sequence):
   
    def __init__(self, X1, X2, clinics, times, events, classificationTask, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X1.shape[1:])
        self.n_subject = X1.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.X1 = X1[sorted_indices]
        self.X2 = X2[sorted_indices]
        self.clinics = clinics[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]
        self.classificationTask = classificationTask[sorted_indices]

        self.indices = range(self.n_subject)

        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X1 = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        X2 = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)

        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        
        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
        clinics = self.clinics[sorted_batch_indices]
        classificationTask = self.classificationTask[sorted_batch_indices]

        for i_batch in range(self.batch_size):
            image_index = sorted_batch_indices[i_batch]
                
            # Data augmentation
            subject_tio = tio.Subject(image1 = tio.ScalarImage(tensor = np.expand_dims(self.X1[image_index], 0)), image2 = tio.ScalarImage(tensor = np.expand_dims(self.X2[image_index], 0)))
            subject_tio_augmented = self.data_augmentation_transform(subject_tio)
            X1[i_batch, :, :, :, 0] = subject_tio_augmented['image1'].numpy()[0]
            X2[i_batch, :, :, :, 0] = subject_tio_augmented['image2'].numpy()[0]

        return [X1, X2, clinics, events, classificationTask], times
        
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1




class Generator_clinics(Sequence):
   
    def __init__(self, clinics, times, events, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.n_subject = clinics.shape[0]

        sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.clinics = clinics[sorted_indices]
        self.times = times[sorted_indices]
        self.events = events[sorted_indices]

        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
      	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        batch_indices = np.array(random.sample(self.indices, self.batch_size))

        while ( np.sum(self.events[batch_indices]) < 3):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        #while ( np.abs(np.sum(self.events[batch_indices]) - (self.batch_size //2) ) > 1e-5):
         # batch_indices = np.array(random.sample(self.indices, self.batch_size))

        sorted_batch_indices = np.sort(batch_indices)

        times = self.times[sorted_batch_indices]
        events = self.events[sorted_batch_indices]
        clinics = self.clinics[sorted_batch_indices]
        #print(clinics)  
     
        return [clinics, events], times
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1

          
class Generator_clinics_OS_PFS(Sequence):
   
    def __init__(self, clinics, times_OS, events_OS, times_PFS, events_PFS, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.n_subject = clinics.shape[0]

        #sorted_indices = np.argsort(-times, axis = 0) # Sort the images, times, and events by descending order

        self.clinics = clinics
        self.times_OS = times_OS
        self.events_OS = events_OS
        self.times_PFS = times_PFS
        self.events_PFS = events_PFS
	
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
      	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        batch_indices = np.array(random.sample(self.indices, self.batch_size))

        while ( (np.sum(self.events_OS[batch_indices]) < 3) or (np.sum(self.events_PFS[batch_indices]) < 3) ):
          batch_indices = np.array(random.sample(self.indices, self.batch_size)) 
        
        
        sorted_index_OS = np.argsort(-self.times_OS[batch_indices], axis = 0)
        sorted_batch_indices_OS = batch_indices[sorted_index_OS]
        sorted_index_PFS = np.argsort(-self.times_PFS[batch_indices], axis = 0)
        sorted_batch_indices_PFS = batch_indices[sorted_index_PFS]
        
        times_OS = self.times_OS[sorted_batch_indices_OS]
        events_OS = self.events_OS[sorted_batch_indices_OS]
        clinics_OS = self.clinics[sorted_batch_indices_OS]

        times_PFS = self.times_PFS[sorted_batch_indices_PFS]
        events_PFS = self.events_PFS[sorted_batch_indices_PFS]
        clinics_PFS = self.clinics[sorted_batch_indices_PFS]

        #print(clinics)  
     
        return [clinics_OS, events_OS, clinics_PFS, events_PFS], [times_OS, times_PFS]
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
 


class Generator3D_segOnly(Sequence):
   
    def __init__(self, X, Y, labels=[1], batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.labels = labels

        self.X = X
        self.Y = Y
        self.indices = range(self.n_subject)

        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.65)

	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], len(self.labels)), dtype= np.float32)

        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        for i_batch in range(self.batch_size):
            image_index = batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)), segmentation = tio.LabelMap(tensor = np.expand_dims(self.Y[image_index], 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            Y[i_batch] = multi_class_labels(patch_tio_augmented['segmentation'].numpy()[0], self.labels)  
            
        return X, Y
        
        
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
 

class Generator3D_segOnly_withoutDataAugmentation(Sequence):
   
    def __init__(self, X, Y, labels=[1], batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.labels = labels

        self.X = X
        self.Y = Y
        self.indices = range(self.n_subject)

        self.step_by_epoch = 300
        self.current_epoch = 0
	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'
       
        Y = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], len(self.labels)), dtype= np.float32)

        batch_indices = np.array(random.sample(self.indices, self.batch_size))
        
        for i_batch in range(self.batch_size):
             Y[i_batch] = multi_class_labels(self.Y[batch_indices[i_batch]], self.labels)

        return self.X[batch_indices], Y
        
        
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1


 
class Generator_CNN3DModel(Sequence):
   
    def __init__(self, X, Y, batch_size=32):

        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        self.X = X
        self.Y = Y
 
        self.indices = range(self.n_subject)
        
        self.step_by_epoch = 300
        self.current_epoch = 0
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(10,10,10), translation = (5, 5, 5), default_pad_value = "mean"): 0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.6}, p = 0.70))
        #self.data_augmentation_transform = tio.RandomAffine(scales = (1,1), degrees=(10, 10, 10), translation = (5, 5, 5), default_pad_value = "mean", p = 0.70)

	
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
        
        batch_indices = np.array(random.sample(self.indices, self.batch_size))       
     
        Y = self.Y[batch_indices]
  
        for i_batch in range(self.batch_size):
            image_index = batch_indices[i_batch]
                
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(self.X[image_index], 0)))   
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)
            X[i_batch, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            
        return X, Y
        
       
    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
        

              
