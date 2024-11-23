import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mobilenet_basemodel import model

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training the parameters
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    'dataset/train',  # TODO: Edit and put in the actual path to our dataset
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Training the model
model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator))