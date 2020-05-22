import pandas as pd
import logging
import argparse
from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils
from imdb_data.face_estimate.source_file.utils import load_data
from imdb_data.face_estimate.source_file.finetune_modle import get_model, age_mae
from imdb_data.face_estimate.source_file.imdb_generator import FaceGenerator, ValGenerator


logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--input", "-i", type=str, required=True,
    #                     help="path to input database mat file")
    parser.add_argument("--input", "-i", type=str, default="G:/Face age estimation/imdb_data/face_estimate/source_file/finetune_with_appa/data_set/1/picture_inf.mat",
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=32,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam' or 'Adagrad'")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    parser.add_argument("--output_path", type=str, default="Checkpoint",
                        help="checkpoint dir")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr
        return self.initial_lr


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    elif opt_name == "Adagrad":
        return Adagrad(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    train_mat = "G:/Face age estimation/appa-real-release/train_finetune_inf.mat"
    valid_mat = "G:/Face age estimation/appa-real-release/valid_finetune_inf.mat"
    full_path_1, age_1 = load_data(train_mat)
    full_path_2, age_2 = load_data(valid_mat)
    train_image_path_and_age = []
    vali_image_path_and_age = []
    args = get_args()
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt

    output_path = Path("G:/Face age estimation/appa-real-release/model")

    logging.debug("Loading data...")


    model = get_model(model_name="VGG16")
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=[age_mae])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_age_mae:.2f}.hdf5",
                                 monitor="val_age_mae",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]

    logging.debug("Running training...")


    for i in range(len(full_path_1)):
        train_image_path_and_age.append([full_path_1[i], age_1[i]])
    for i in range(len(full_path_2)):
        vali_image_path_and_age.append([full_path_2[i], age_2[i]])

    train_gen = FaceGenerator(appa_dir=train_image_path_and_age, utk_dir=None, batch_size=batch_size, image_size=224)
    val_gen = ValGenerator(appa_dir=vali_image_path_and_age, batch_size=batch_size, image_size=224)
    hist = model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)
    logging.debug("Saving history...")
    pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history.h5"), "history")


if __name__ == '__main__':
    main()
