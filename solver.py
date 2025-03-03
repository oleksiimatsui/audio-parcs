from Pyro4 import expose
import wave
import struct
import time

def toFloat(samples):
    return [s / 32767.0 for s in samples]

def toInt(samples):
    return [int(s * 32767) for s in samples]

def getFormat(bytes):
    return {1: "b", 2: "h", 4: "i", 8: "q"}.get(bytes, None)

def process(last_samples, samples, delay, decay=0.5, framerate=44100):
    length = len(samples)
    output = [0] * (length + delay)

    for i in range(min(delay, len(last_samples))):
        output[i] += last_samples[i] * decay

    for i in range(length - delay):
        output[i] += samples[i]
        output[i + delay] += samples[i] * decay
    
    for i in range(length - delay, length):
        output[i] += samples[i]

    return output

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers or []
        print("Inited")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))
        samples = self.read_input()

        print("File is loaded")
        step = len(samples) // len(self.workers)
        delay_ms = 50
        framerate = 44100    
        delay = int((delay_ms / 1000) * framerate)

        start_time = time.time()
        # map
        mapped = []
        mapped.append(self.workers[0].mymap([0] * (delay * 2), samples[0:step], delay, decay=0.8))
        for i in range(1, len(self.workers)):
            print(i)
            mapped.append(self.workers[i].mymap(
                samples[i * step - delay * 2: i * step],
                samples[i * step:i * step + step],
                delay
            ))

        print('Map finished')
        # reduce
        reduced = self.myreduce( [res.value for res in mapped] )
        print("Reduce finished")
        
        print("--- %s seconds ---" % (time.time() - start_time))
 
        # output
        self.write_output([int(x) for x in reduced])
        print("Job Finished")

    @staticmethod
    @expose
    def mymap(last_samples, samples, delay, decay=0.5, framerate=44100):
        last_samples = toFloat(last_samples)
        samples = toFloat(samples)
        left = process(last_samples[::2], samples[::2], delay, decay=decay, framerate=framerate)
        right = process(last_samples[1::2], samples[1::2], delay, decay=decay, framerate=framerate)
        
        res = [val for pair in zip(left, right) for val in pair]
        max_val = max(abs(x) for x in res)
        return res, max_val

    @staticmethod
    @expose
    def myreduce(mapped):
        mapped_samples = []
        max_sample = max(m[1] for m in mapped)
        
        for m in mapped:
            mapped_samples.extend(m[0])
        
        return toInt([x / max_sample for x in mapped_samples])
    
    def read_input(self):
        wav_file = wave.open(self.input_file_name, "rb")
        try:
            sample_width = wav_file.getsampwidth()
            raw_data = wav_file.readframes(wav_file.getnframes())
            return list(struct.unpack("<" + getFormat(sample_width) * (len(raw_data) // sample_width), raw_data))
        finally:
            wav_file.close()

    def write_output(self, samples):
        wav_file = wave.open(self.output_file_name, "w")
        try:
            nchannels, sampwidth, framerate = 2, 2, 44100
            nframes = len(samples) // nchannels
            wav_file.setparams((nchannels, sampwidth, framerate, nframes, "NONE", "not compressed"))
            wav_file.writeframes(struct.pack("<" + "h" * len(samples), *samples))
        finally:
            wav_file.close()  # Explicitly close the file
# for testing
# s = Solver(input_file_name="./data/song.wav", output_file_name="./new.wav")
# s.workers = [s]
# s.solve()
