import better_exceptions
from keras.applications import VGG16, ResNet50
from keras.layers import Dense
from keras.models import Model
from keras import backend as K


weight_file = "G:/Face age estimation/imdb_data/face_estimate/source_file/Finetune_with_imdb_second/weights.20-6.96_7.21.hdf5"


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="VGG16",trainable = None):
    base_model = None

    if model_name == "VGG16":
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")

    elif model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
        if trainable:
            set_trainable = False
            for layer in base_model.layers:
                if layer.name == 'res5c_branch2a':
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False



    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)

    print('num of trainable weights before freezing the conv base:', len(model.trainable_weights))
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    #
    # print('num of trainable weights after freezing the conv base:', len(model.trainable_weights))

    return model


def main():
    model = get_model("VGG16")
    model.summary()


if __name__ == '__main__':
    main()
