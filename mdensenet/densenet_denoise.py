from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Concatenate, MaxPooling2D
from keras.models import Model
import numpy as np
import os
import console
import conversion
from data import Data
import argparse
import random
import string


# Network parameters
class Densenet_denoise:
    def __init__(self):
        # Input
        noisy = Input(shape=(None, None, 1), name='input')

        # dense block 1
        convA = Conv2D(32, 3, activation='relu', padding='same')(noisy)  # 128*128*64
        convA = Concatenate()([noisy, convA])   # 128*128*65
        convA = Conv2D(32, 3, activation='relu', padding='same')(convA)    # 128*128*64
        convA = BatchNormalization()(convA)

        # down sample 1
        ds1 = Conv2D(32, 4, strides=2, activation='relu', padding='same', use_bias=False)(convA)  # 64*64*64

        # dense block 2
        convB = Conv2D(64, 3, activation='relu', padding='same')(ds1)  # 64*64*64
        convB = Concatenate()([ds1, convB])   # 128*128*128
        convB = Conv2D(64, 3, activation='relu', padding='same', use_bias=False)(convB)    # 64*64*64
        convB = BatchNormalization()(convB)

        # down sample 2
        ds2 = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(convB)  # 32*32*64

        # dense block 3
        convC = Conv2D(128, 3, activation='relu', padding='same')(ds2)  # 32*32*128
        convC = Concatenate()([ds2, convC])  # 32*32*192
        convC = Conv2D(128, 3, activation='relu', padding='same')(convC)  # 32*32*128
        convC = BatchNormalization()(convC)

        # up sample 1
        up1 = UpSampling2D((2, 2))(convC)  # 64*64*128

        # dense block 4
        conv = Concatenate()([up1, convB])  # 64*64*192
        convD = Conv2D(64, 3, activation='relu', padding='same')(conv)  # 64*64*64
        convD = Concatenate()([conv, convD])  # 64*64*256
        convD = Conv2D(64, 3, activation='relu', padding='same')(convD)  # 64*64*64
        convD = BatchNormalization()(convD)

        # up sample 2
        up2 = UpSampling2D((2, 2))(convD)  # 128*128*64

        # dense block 5
        conv = Concatenate()([up2, convA])  # 128*128*128
        convE = Conv2D(32, 3, activation='relu', padding='same')(conv)  # 128*128*64
        convE = Concatenate()([conv, convE])  # 128*128*192
        convE = Conv2D(32, 3, activation='relu', padding='same')(convE)  # 128*128*64
        convE = BatchNormalization()(convE)

        # fully connection
        clean = Conv2D(32, 3, activation='relu', padding='same')(convE)
        clean = Conv2D(32, 3, activation='relu', padding='same')(clean)  # 128*128*32
        clean = Conv2D(1, 3, activation='relu', padding='same')(clean)  # 128*128*1

        dense_net = Model(inputs=noisy, outputs=clean)
        # dense_net.summary()
        print("Model has", dense_net.count_params(), "params")
        dense_net.compile(loss='mean_squared_error', optimizer='rmsprop')
        self.model = dense_net
        # need to know so that we can avoid rounding errors with spectrogram
        # this should represent how much the input gets downscaled
        # in the middle of the network
        self.peakDownscaleFactor = 4

    def train(self, data, epochs, batch=8):
        xTrain, yTrain = data.train()
        xValid, yValid = data.valid()
        while epochs > 0:
            console.log("Training for", epochs, "epochs on", len(xTrain), "examples")
            self.model.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))
            console.notify(str(epochs) + " Epochs Complete!", "Training on", data.inPath, "with size", batch)
            while True:
                try:
                    epochs = int(input("How many more epochs should we train for? "))
                    break
                except ValueError:
                    console.warn("Oops, number parse failed. Try again, I guess?")
            if epochs > 0:
                save = input("Should we save intermediate weights [y/n]? ")
                if not save.lower().startswith("n"):
                    weightPath = ''.join(random.choice(string.digits) for _ in range(16)) + ".h5"
                    console.log("Saving intermediate weights to", weightPath)
                    self.saveWeights(weightPath)

    def saveWeights(self, path):
        self.model.save_weights(path, overwrite=True)

    def loadWeights(self, path):
        self.model.load_weights(path)

    def isolateVocals(self, path, fftWindowSize, phaseIterations=10):
        console.log("Attempting to isolate vocals from", path)
        audio, sampleRate = conversion.loadAudioFile(path, sr=22100)
        spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize=fftWindowSize)
        console.log("Retrieved spectrogram; processing...")

        expandedSpectrogram = conversion.expandToGrid(spectrogram, self.peakDownscaleFactor)
        expandedSpectrogramWithBatchAndChannels = expandedSpectrogram[np.newaxis, :, :, np.newaxis]
        print(expandedSpectrogramWithBatchAndChannels.shape)
        # 预测
        predictedSpectrogramWithBatchAndChannels = self.model.predict(expandedSpectrogramWithBatchAndChannels)

        predictedSpectrogram = predictedSpectrogramWithBatchAndChannels[0, :, :, 0]  # o /// o
        newSpectrogram = predictedSpectrogram[:spectrogram.shape[0], :spectrogram.shape[1]]
        console.log("Processed spectrogram; reconverting to audio")
        newAudio = conversion.spectrogramToAudioFile(newSpectrogram, sampleRate, fftWindowSize=fftWindowSize, phaseIterations=phaseIterations)

        # file information
        pathParts = os.path.split(path)
        fileNameParts = os.path.splitext(pathParts[1])
        outputFileNameBase = os.path.join(pathParts[0], fileNameParts[0] + "_densenet")
        console.log("Converted to audio; writing to", outputFileNameBase)

        conversion.saveAudioFile(newAudio, outputFileNameBase + ".wav", sampleRate)
        # conversion.saveSpectrogram(newSpectrogram, outputFileNameBase + ".png")
        # conversion.saveSpectrogram(spectrogram, os.path.join(pathParts[0], fileNameParts[0]) + ".png")
        console.log("Vocal isolation complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--data_path", default='train_data', type=str, help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float, help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default='denoise_v1_3.h5', type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=50, type=int, help="Batch size for training")
    parser.add_argument("--phase", default=10, type=int, help="Phase iterations for reconstruction")
    parser.add_argument("--load", default=False, action='store_true', help="Load previous weights file before starting")
    parser.add_argument("files", nargs="*", default=[])
    args = parser.parse_args()

    densenet_denoise = Densenet_denoise()

    print(len(args.files))
    if len(args.files) == 0 and args.data_path:
        console.log("No files provided; attempting to train on " + args.data_path + "...")
        if args.load:
            console.h1("Loading Weights")
            densenet_denoise.loadWeights(args.weights)
        console.h1("Loading Data")
        data = Data(args.data_path, args.fft, args.split)
        console.h1("Training Model")
        densenet_denoise.train(data, args.epochs, args.batch)
        densenet_denoise.saveWeights(args.weights)
    elif len(args.files) > 0:
        console.log("Weights provided; performing inference on " + str(args.files) + "...")
        console.h1("Loading weights")
        densenet_denoise.loadWeights(args.weights)
        for f in args.files:
            densenet_denoise.isolateVocals(f, args.fft, args.phase)
    else:
        console.error("Please provide data to train on (--data) or files to infer on")

