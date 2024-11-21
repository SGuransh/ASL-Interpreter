import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Model

# Loading the base MobileNetV2 model, without the top (classification) layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

### TODO: EDIT WEIGHTS ###
# Adding custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation='relu')(x)  # Fully-connected layer for higher-level features
predictions = Dense(27, activation='softmax')(x)  # 27 output classes (26 letters + 1 blank space)
#########################

# Creating the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Freezing all layers in the base model to keep pretrained weights
for layer in base_model.layers:
    layer.trainable = False

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', # TODO: Edit loss function if required
              metrics=['accuracy'])

# Printing the model summary
model.summary()


### TODO: HOW TO UNFREEZE LAYERS TO POTENTIALLY IMPROVE ACCURACY ###
# After training the custom layers for some epochs, you may unfreeze some of the earlier layers in MobileNetV2 and train the entire network at a lower learning rate for improved accuracy:
# # Unfreeze some layers in the base model
# for layer in base_model.layers[-30:]:
#     layer.trainable = True

# # Compile again with a lower learning rate for fine-tuning
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#########################
