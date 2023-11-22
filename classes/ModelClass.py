from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt
import visualkeras
from PIL import ImageFont
from keras.optimizers import Adam
from PIL import ImageFont
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
os.listdir()

# Input de imagen
IMAGE_SIZE = [224, 224]

# Inicialización arquitectura VGG16
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

for layer in vgg.layers:
    layer.trainable = False

folders = glob("./faces/train/*")

# Implementación de clase Model de Keras
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs = vgg.input, outputs = prediction)

# Graficación de arquitectura de modelo en 2D
# Nota: visual keras solamente sirve con la versión 9.5.0 de Pillow
font = ImageFont.truetype("arial.ttf", 50)
visualkeras.layered_view(model, to_file='model2D1.png', spacing=2, legend=True, font=font)

# Resumen del modelo
print(model.summary())

# Compilación de modelo
model.compile(
        loss = 'categorical_crossentropy',
        optimizer = Adam(learning_rate=0.0015),
        metrics = ['accuracy']
        )

# Preprocesamiento de imágenes
train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range= 0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale= 1./255)

training_set = train_datagen.flow_from_directory("./faces/train",
                                                       target_size= (224,224),
                                                       class_mode='categorical')

test_set = test_datagen.flow_from_directory("./faces/validation",
                                            target_size= (224,224),
                                            class_mode= 'categorical')

print(training_set)
print(test_set)

# Evitar sobreajuste
#early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
#mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Entrenamiento del modelo
r = model.fit(
        training_set,
        validation_data= test_set,
        epochs=5,
        steps_per_epoch= len(training_set),
        validation_steps=len(test_set)
        )

# Gráfico de pérdida
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig("VGG16_train-validation_loss60-40.png") 
plt.show()

# Gráfico de precisión
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig("VGG16_train-validation_accuracy60-40.png") 
plt.show()

# Guardado de modelo
model.save('./models/VGG16_Final_Model_Face.keras')
print('Successfully saved')