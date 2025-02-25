from Pyro4 import expose
import wave
from collections import Counter
import struct
import sys
import scipy.io.wavfile
import numpy as np
from scipy.signal import lfilter
import time

def toFloat(samples):
    return np.array([ np.float32(s/(32768.0)) for s in samples])
def toInt(samples):
    return np.array([ int(s*32768) for s in samples])

def getFormat(bytes):
    fmt = { 1: "b", 2: "h", 4: "i", 8: "q"}.get(bytes, None)
    return fmt

def process(last_samples, samples, delay, decay=0.5, framerate=44100):
        length = len(samples)
        output = np.zeros(length+delay)

        for i in range(delay):
            output[i] += (last_samples[i] * decay)

        for i in range(length-delay):
            output[i] += (samples[i])
            output[i + delay] += (samples[i] * decay); 
        
        for i in range(length, length-delay):
            output[i] += (samples[i])

        return output


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))
        samples = self.read_input()

        print("file is loaded")
        step = samples.size // len(self.workers)
        delay_ms=50
        framerate=44100    
        delay = int((delay_ms / 1000) * framerate)

        start_time = time.time()
        # map
        mapped = []
        mapped.append(self.workers[0].mymap(np.zeros(delay*2), samples[0:step], delay))

        for i in range(1, len(self.workers)):
            print(i)
            mapped.append(self.workers[i].mymap(
                 samples[i*step-delay*2: i*step],
                   samples[i*step:i*step+step],
                   delay ))

        print('Map finished: ', mapped)
        # reduce
        reduced = self.myreduce(mapped)
        print("Reduce finished: " + str(reduced))

        print("--- %s seconds ---" % (time.time() - start_time))
        # output
        self.write_output(reduced.astype(np.int16))
        print("Job Finished")

    @staticmethod
    @expose
    def mymap(last_samples, samples, delay, decay=0.5, framerate=44100):
        # Reshape to (samples, 2)
        last_samples = toFloat(last_samples.reshape(-1, 2))
        samples = toFloat(samples.reshape(-1,2))
        left = process(last_samples[:,0], samples[:,0], delay, decay=0.5, framerate=44100)
        right = process(last_samples[:,1], samples[:,1], delay, decay=0.5, framerate=44100)
        res =  np.ravel(np.column_stack((left, right)))
        max = np.max(np.abs(res))
        return res, max
    
    

    @staticmethod
    @expose
    def myreduce(mapped):
        mappedSamples = list()
        maxSample = 0
        for i in range(len(mapped)):
            maxSample = max(maxSample, mapped[i][1])
        for i in range(len(mapped)):
            mappedSamples.append(mapped[i][0])
        return toInt( np.concatenate( mappedSamples/ maxSample) )
    

    def read_input(self):
        with wave.open(self.input_file_name, "rb") as wav_file:
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(num_frames)
                samples = list(struct.unpack("<" + getFormat(sample_width) * (len(raw_data) // sample_width), raw_data))
                return np.array(samples)

    def write_output(self, samples):
            wav_file = wave.open(self.output_file_name, "w")
            nchannels = 2
            sampwidth = 2
            size = len(samples)
            nframes = size // nchannels 
            comptype = "NONE"
            compname = "not compressed"
            wav_file.setparams((nchannels, sampwidth, 44100, nframes, comptype, compname))
            array = struct.pack("<" + "h" * size, *samples)
            wav_file.writeframes(array)
            wav_file.close()




# for testing
s = Solver(input_file_name="./data//audio.wav", output_file_name="./new.wav")
s.workers = [s]
s.solve()