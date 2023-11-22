
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import visualkeras
from PIL import ImageFont
from keras.optimizers import Adam
from PIL import ImageFont
import os
os.listdir()

# Input de imagen
IMAGE_SIZE = [150, 150]

# Inicialización arquitectura VGG16
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

for layer in vgg.layers:
    layer.trainable = False

folders = glob("./faces/train/*")

# Implementación de modelo Sequential
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(len(folders), activation='softmax'))

# Graficación de arquitectura de modelo en 2D
# Nota: visual keras solamente sirve con la versión 9.5.0 de Pillow
font = ImageFont.truetype("arial.ttf", 50)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, to_file='Sequential_model2D.png', spacing=2, legend=True, font=font)

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
                                                       target_size= (150, 150),
                                                       class_mode='categorical')

test_set = test_datagen.flow_from_directory("./faces/validation",
                                            target_size= (150, 150),
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
        epochs=20,
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
plt.savefig("Sequential_train-validation_loss.png") 
plt.show()

# Gráfico de precisión
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig("Sequential_train-validation_accuracy.png") 
plt.show()

# Guardado de modelo
model.save('./models/Sequential_Final_Model_Face.keras')
print('Successfully saved')