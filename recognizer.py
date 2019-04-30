import librosa
import numpy as np
import pyaudio
import wave
from dtw import dtw
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal

#function to record new sound
def record(WAVE_OUTPUT_FILENAME):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 2

    device_index = 2
    audio = pyaudio.PyAudio()
    cutoff = 3000
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    print("-------------------------------------------------------------")

    index = (input())
    print("recording via index " + str(index))

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=int(index),
                        frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    print('\n'*5)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()


#z transform sound
def normalize(sound):
    s = sound.copy()
    mean = np.mean(sound)
    std = np.std(sound)
    for i in range(len(sound)):
        s[i] = float((sound[i] - mean) / std)
    return s

#clear from noize: only frequences 300-3000
def noise_reduction(wavFile):
    (Frequency, samples) = read(wavFile)
    b, a = signal.butter(5, 300 / Frequency, btype='highpass')
    filteredSignal = signal.lfilter(b, a, samples)
    c, d = signal.butter(5, 3000 / Frequency, btype='lowpass')
    newFilteredSignal = signal.lfilter(c, d, filteredSignal)
    write(wavFile + 'RED', Frequency, newFilteredSignal)
    return Frequency, normalize(newFilteredSignal)


def compare(mfсс1, mfсс2):
    dists = []
    if len(mfсс1) >= len(mfсс2):
        return 0
    window_size = len(mfсс1)
    for i in range(len(mfсс2) - window_size):
        d, a, b, c = dtw(mfсс1, mfсс2[i:i + window_size], dist=lambda x, y: np.linalg.norm(x - y))
        dists.append(d)
    return min(dists), dists

def show_signa(sound1, fs1, sound2, fs2):
    mfc1 = librosa.feature.mfcc(sound1, fs1, n_mfcc=20)
    mfc2 = librosa.feature.mfcc(sound2, fs2, n_mfcc=20)
    dist, dists = compare(mfc1.T, mfc2.T)
    print('dtw distance {}'.format(round(dist)))
    if dist <= 30:
        print('I recognized your word!!!')
        part = (dists.index(dist) + 1) / len(dists) -0.1
        s2 = sound2[int(round(len(sound2) * part)):int(round(len(sound2) * part)) + len(sound1)]
        plt.plot(sound2)
        plt.axvspan(int(round(len(sound2) * part)), int(round(len(sound2) * part)) + len(sound1), color='red',
                alpha=0.5)
        plt.show()
    else:
        print('you did not say train word')

#helps to find your word in silence
def find_wave(sound):
    begin = None
    end = None
    for num, i in enumerate(sound):
        if i >=2:
            begin = num
            break
    for num, i in enumerate(sound[::-1]):
        if i >=2:
            end = len(sound) - num
            break
    s = sound[begin:end].copy()
    plt.plot(s)
    plt.show()
    return s

#find distance between two mfccs
def dist(self, sign1, f1, sign2, f2):
    mfc1 = librosa.feature.mfcc(sign1, f1, n_mfcc=20)
    mfc2 = librosa.feature.mfcc(sign2, f2, n_mfcc=20)
    d, a, b, c = dtw(mfc1.T, mfc2.T, dist=lambda x, y: np.linalg.norm(x - y))
    return d

class Audio:
    def __init__(self, valid, totest):
        self.validFreq, self.validSound = noise_reduction(valid)
        self.validSound = find_wave(self.validSound)
        self.testFreq, self.testSound = noise_reduction(totest)