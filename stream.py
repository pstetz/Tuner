"""
Imports
"""
import pyaudio
import numpy as np

"""
Stream functions
"""
def _open_stream(config):
    """ Opens a pyaudio stream that's needed to record """
    p = pyaudio.PyAudio()
    stream = p.open(
        format = config["format"],
        channels = config["channels"],
        rate = config["rate"],
        input = True,
        frames_per_buffer = config["chunk"]
    )
    return p, stream

def _record_wav(stream, N, CHUNK):
    """ Records N times in units of CHUNK """
    frames = []
    for i in range(N):
        data = stream.read(CHUNK)
        frames.append(data)
    return np.fromstring(b"".join(frames), 'Int16')

def _close_stream(p, stream):
    """ Close the pyaudio stream """
    stream.stop_stream()
    stream.close()
    p.terminate()

def sample(seconds, stream_config):
    """ Samples a recording """
    ### Open a stream
    p, stream = _open_stream(stream_config)

    ### Get sampling rate and CHUNK from the stream
    chunk = stream_config["chunk"]
    sampling_rate = stream_config["rate"]

    ### Sample
    N = int(seconds / (chunk / sampling_rate))
    data = _record_wav(stream, N, chunk)

    ### Close the stream
    _close_stream(p, stream)
    return data

