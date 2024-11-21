import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the PaliGemma model from TensorFlow Hub (hypothetical link)
import tensorflow_hub as hub

# Loading the PaliGemma base model without top layers
base_model = hub.KerasLayer("https://tfhub.dev/paligemma/paligemma-v1/feature-vector", 
                            input_shape=(224, 224, 3), trainable=False)

### TODO: EDIT WEIGHTS ###
# Adding custom layers for classification on top of the base model
x = base_model.output if hasattr(base_model, 'output') else base_model
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation='relu')(x)  # Fully-connected layer for higher-level features
predictions = Dense(29, activation='softmax')(x)  # 27 output classes (26 letters + 1 blank space)
#########################

# Creating the complete model
model = Model(inputs=base_model.inputs, outputs=predictions)

# Freezing all layers in the base model to keep pretrained weights
base_model.trainable = False

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Printing the model summary
model.summary()

## TODO: HOW TO UNFREEZE LAYERS TO POTENTIALLY IMPROVE ACCURACY ###
# # Unfreeze all layers to fine-tune the entire model
# base_model.trainable = True

# # Compile again with a lower learning rate
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#########################
