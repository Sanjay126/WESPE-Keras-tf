
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from datagen import DataGenerator as DataGenerator2
import datagen
from tensorflow.keras.applications.vgg_19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split



class VGG_19():

    HEIGHT = 224
    WIDTH = 224

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

    def build_finetune_model(base_model, dropout, fc_layers, num_classes):
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        for fc in fc_layers:
            # New FC layer, random init
            x = Dense(fc, activation='relu')(x) 
            x = Dropout(dropout)(x)

        # New softmax layer
        predictions = Dense(num_classes, activation='softmax')(x) 
        
        finetune_model = Model(inputs=base_model.input, outputs=predictions)

        return finetune_model

    class_list = ["Original","Tampered"]
    FC_LAYERS = [1024, 1024]
    NUM_EPOCHS = 100
    BATCH_SIZE = 25
    lr=0.00005
    dropout = 0.5

    finetune_model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))

    adam = Adam(lr=0.00001)
    finetune_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

    filepath="./checkpoints/" + "MobileNetV2" + "_model_weights2.h5"
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                                           shuffle=True, callbacks=callbacks_list,validation_data=valid_generator,validation_freq=5,use_multiprocessing=True)


