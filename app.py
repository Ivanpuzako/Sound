from recognizer import Audio,  show_signa, record

TRAIN_WAVE_OUTPUT_FILENAME = 'train.wav'
WAVE_OUTPUT_FILENAME = "recordedFile.wav"

print('1.Record new sound (2 seconds) and train it to recognize your word in the next recording')
print('say your word in silence')
record(TRAIN_WAVE_OUTPUT_FILENAME)
print('say something to recognize your word')
record(WAVE_OUTPUT_FILENAME)
recognizer = Audio(TRAIN_WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILENAME)

print('--------------------------please wait-------------------------------------')
res = show_signa(recognizer.validSound, recognizer.validFreq, recognizer.testSound, recognizer.testFreq)