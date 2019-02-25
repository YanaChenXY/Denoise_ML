"""
Loads and stores mashup data given a folder full of acapellas and instrumentals
Assumes that all audio clips (wav, mp3) in the folder
a) have their Camelot key as the first token in the filename
b) are in the same BPM
c) have "acapella" somewhere in the filename if they're an acapella, and are otherwise instrumental
d) all have identical arrangements
e) have the same sample rate
"""
import sys
import os
import numpy as np
import h5py

import console
import conversion

from math import ceil


# Modify these functions if your data is in a different format
def keyOfFile(fileName):
    firstToken = int(fileName.split('_')[0])
    if 0 < firstToken <= NUMBER_OF_KEYS:
        return firstToken
    console.warn("File", fileName, "doesn't specify its key, ignoring..")
    return None


def fileIsAcapella(fileName):
    return "acapella" in fileName.lower()


NUMBER_OF_KEYS = 12  # number of keys to iterate over
SLICE_SIZE = 128  # size of spectrogram slices to use


# Slice up matrices into squares so the neural net gets a consistent size for training (doesn't matter for inference)
# def chop(matrix, scale):
#     slices = []
#     for time in range(0, matrix.shape[1] // scale):
#         for freq in range(0, matrix.shape[0] // scale):
#             s = matrix[freq * scale: (freq + 1) * scale,
#                 time * scale: (time + 1) * scale]
#             slices.append(s)
#     return slices


def chop(matrix, scale):
    slices = []
    for time in range(0, ceil(matrix.shape[1] / scale)):
        if matrix.shape[1] < (time + 1) * scale:
            next_time = matrix.shape[1]
            y = next_time % scale
        else:
            next_time = (time + 1) * scale
            y = scale
        for freq in range(0, ceil(matrix.shape[0] / scale)):
            if matrix.shape[0] < (freq + 1) * scale:
                next_freq = matrix.shape[0]
                x = next_freq % scale
            else:
                next_freq = (freq + 1) * scale
                x = scale
            s = np.zeros((scale, scale))
            s[:x, :y] = matrix[freq * scale: next_freq, time * scale:next_time]
            slices.append(s)
    return slices


class Data:
    def __init__(self, inPath, fftWindowSize=1536, trainingSplit=0.9):
        self.inPath = inPath
        self.fftWindowSize = fftWindowSize
        self.trainingSplit = trainingSplit
        self.rate=0.5
        self.x = []
        self.y = []
        self.load()

    def train(self):
        return self.x[:int(len(self.x) * self.trainingSplit*self.rate)], self.y[:int(len(self.y) * self.trainingSplit*self.rate)]

    def valid(self):
        return self.x[int(len(self.x) * self.trainingSplit*self.rate):int(len(self.x)*self.rate)], self.y[int(len(self.y)*self.trainingSplit*self.rate):int(len(self.x)*self.rate)]

    def load(self, saveDataAsH5=True):
        h5Path = os.path.join(self.inPath, "data.h5")
        print(h5Path)
        # h5Path = '/Users/Yana/Documents/myself_doc/008_Python_Project/acapellabot/train_data/data.h5'
        if os.path.isfile(h5Path):
            h5f = h5py.File(h5Path, "r")
            self.x = h5f["x"][:]
            self.y = h5f["y"][:]
        else:
            # Hash bins for each camelot key so we can merge
            # in the future, this should be a generator w/ yields in order to eat less memory

            count = 0

            for dirPath, dirNames, fileNames in os.walk(os.path.join(self.inPath, 'noisy')):
                for fileName in filter(lambda f: (f.endswith(".mp3") or f.endswith(".wav")) and not f.startswith("."),
                                       fileNames):
                    audio, sampleRate = conversion.loadAudioFile(os.path.join(self.inPath, 'noisy', fileName))
                    noisy_spectrogram, _ = conversion.audioFileToSpectrogram(audio, self.fftWindowSize)

                    audio, sampleRate = conversion.loadAudioFile(os.path.join(self.inPath, 'clean', fileName))
                    clean_spectrogram, _ = conversion.audioFileToSpectrogram(audio, self.fftWindowSize)

                    if noisy_spectrogram.shape[1] < clean_spectrogram.shape[1]:
                        newInstrumental = np.zeros(clean_spectrogram.shape)
                        newInstrumental[:noisy_spectrogram.shape[0], :noisy_spectrogram.shape[1]] = noisy_spectrogram
                        noisy_spectrogram = newInstrumental
                    elif clean_spectrogram.shape[1] < noisy_spectrogram.shape[1]:
                        newAcapella = np.zeros(noisy_spectrogram.shape)
                        newAcapella[:clean_spectrogram.shape[0], :clean_spectrogram.shape[1]] = clean_spectrogram
                        clean_spectrogram = newAcapella
                    # simulate a limiter/low mixing (loses info, but that's the point)
                    # I've tested this against making the same mashups in Logic and it's pretty close
                    # mashup = np.maximum(clean_spectrogram, noisy_spectrogram)
                    # AttributeError: 'list' object has no attribute 'shape'chop into slices so everything's the same size in a batch
                    dim = SLICE_SIZE
                    mashupSlices = chop(noisy_spectrogram, dim)
                    acapellaSlices = chop(clean_spectrogram, dim)
                    count += 1
                    if acapellaSlices.__len__()>0:
                        self.x.extend(mashupSlices)
                        self.y.extend(acapellaSlices)
                        console.info(count, "Created spectrogram for", fileName,  "with length", acapellaSlices.__len__())

            # Add a "channels" channel to please the network
            self.x = np.array(self.x)[:, :, :, np.newaxis]
            self.y = np.array(self.y)[:, :, :, np.newaxis]

            console.info('Train data shape: x: ', self.x.shape, '   y: ', self.y.shape)
            # Save to file if asked
            if saveDataAsH5:
                h5f = h5py.File(h5Path, "w")
                h5f.create_dataset("x", data=self.x)
                h5f.create_dataset("y", data=self.y)
                h5f.close()


if __name__ == "__main__":
    # Simple testing code to use while developing
    console.h1("Loading Data")
    d = Data(sys.argv[1], 1536)
    console.h1("Writing Sample Data")
    conversion.saveSpectrogram(d.x[0], "x_sample_0.png")
    conversion.saveSpectrogram(d.y[0], "y_sample_0.png")
    audio = conversion.spectrogramToAudioFile(d.x[0], 1536)
    conversion.saveAudioFile(audio, "x_sample.wav", 22050)
