from keras.layers import Dense
from keras import layers, Model
from keras.applications import ResNet50, VGG16

def get_model(model_name="VGG16"):

    if model_name is "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    else :
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    def model():
        base = base_model
        prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax"
                           )(base_model.output)
        model = Model(inputs=base.input, outputs=prediction)
        return model

    model_1 = model()
    model_2 = model()
    model_3 = model()
    model_4 = model()
    model_5 = model()
    model_6 = model()
    model_7 = model()
    model_8 = model()
    model_9 = model()
    model_10 = model()
    model_11 = model()
    model_12 = model()
    model_13 = model()
    model_14 = model()
    model_15 = model()
    model_16 = model()
    model_17 = model()
    model_18 = model()
    model_19 = model()
    model_20 = model()

    x_1 = model_1.output
    x_2 = model_2.output
    x_3 = model_3.output
    x_4 = model_4.output
    x_5 = model_5.output
    x_6 = model_6.output
    x_7 = model_7.output
    x_8 = model_8.output
    x_9 = model_9.output
    x_10 = model_10.output
    x_11 = model_11.output
    x_12 = model_12.output
    x_13 = model_13.output
    x_14 = model_14.output
    x_15 = model_15.output
    x_16 = model_16.output
    x_17 = model_17.output
    x_18 = model_18.output
    x_19 = model_19.output
    x_20 = model_20.output

    Y = layers.Concatenate(axis=1)([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10,x_11
                                    , x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20])

    end_model = Model(inputs=[base_model.input,], outputs=Y)

    return end_model


if __name__ == '__main__':
    model = get_model("VGG16")
    model.summary()
