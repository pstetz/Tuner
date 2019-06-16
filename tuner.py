"""
Imports
"""

### Main imports
import pyaudio
import numpy as np

### Visualizations
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

### Custom
from info import freq_to_notes
from stream import sample


"""
Data processing
"""
def process_fft(signal, config):
    """ Create and filter the fourier transform signal """
    config_args = ["noise_cutoff", "range_min", "range_max"]
    assert all([arg in config for arg in config_args]), "Expected the following args as input: %s" % ", ".join(config_args)

    ### set variables
    range_min = config["range_min"]
    range_max = config["range_max"]
    noise_cutoff = config["noise_cutoff"]

    fft_signal = np.fft.fft(signal)
    fft_signal[:range_min] = 0
    fft_signal[range_max:] = 0
    fft_signal[fft_signal < noise_cutoff] = 0

    ### Constant height
    max_signal = np.max(fft_signal)
    if max_signal > 0:
        fft_signal = np.true_divide(fft_signal, max_signal)
    return fft_signal

def find_peak(fft_signal):
    return np.argmax(fft_signal)

def find_closest_note(frequency):
    smallest_diff = 7902.13 # frequency of highest pitch note
    closest_note = None
    for freq in freq_to_notes.keys():
        diff = abs(frequency - freq)
        if diff < smallest_diff:
            smallest_diff = diff
            closest_note  = freq
    assert closest_note is not None, "Closest note was not found"
    return closest_note

def determine_note(freq):
    note = freq_to_notes[freq]
    name = note["note"]
    if note["alter"] == 1:
        sign = "#"
    elif note["alter"] == 0:
        sign = ""
    elif note["alter"] == -1:
        sign = "♭"
    else:
        raise Exception("Note definition is incorrect for %s" % str(note))
    return name + sign

def determine_relative_pitch(closest, peak, threshold=0.3):
    if abs(closest - peak) < threshold:
        return "On pitch with %s"
    elif closest - peak < 0:
        return "Flat of %s"
    return "Sharp of %s"


"""
Visualization functions
"""
def plot_sample(signal, fft_config, plot_config):
    """ Displays the raw signal on the left and the fourier transform on the right """
    fig, axarr = plt.subplots(1, 2, figsize=plot_config["figsize"])

    ### Raw signal
    x = np.linspace(0, plot_config["seconds"], len(signal))
    axarr[0].plot(x, signal)
    axarr[0].set_xlabel("Seconds")
    axarr[0].set_title("Raw data")

    ### Fourier transform signal
    # np.clip(fft_signal, range_min, range_max)
    fft_signal = process_fft(signal, fft_config)
    axarr[1].plot(fft_signal)
    axarr[1].set_title("Fourier transform")
    axarr[1].set_xlim(fft_config["range_min"], fft_config["range_max"])

    ### Plot peak pitch
    peak = find_peak(fft_signal)
    axarr[1].axvline(peak, color='black', linewidth=1)
    clear_output()
    plt.show()


"""
Configuration
"""
SECONDS = 0.3
ITERATIONS = 20
stream_config = {
    "chunk": 1024,
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 44100
}
fft_config = {
    "noise_cutoff": 100000,
    "range_min": 12,   # c0 is 16.35 Hz
    "range_max": 1000, # b8 is 7902.13 Hz
}
plot_config = {
    "seconds": SECONDS,
    "figsize": (14, 5),

    "RAW_LINEWIDTH": 0.2,
    "FFT_LINEWIDTH": 0.7,
    "PEAK_LINEWIDTH": 1,

    "Y_LIM_MAX": 1.5,
    "X_TEXT_FREQ": 1,
    "Y_TEXT_FREQ": 1.4,
    "X_TEXT_NOTE": 1,
    "Y_TEXT_NOTE": 1.1,
    "X_TEXT_CLOSEST": 1,
    "Y_TEXT_CLOSEST": 1.25,
}
freq_message    = "Strongest frequency: %s"
closest_message = "Closest frequency: %s"


"""
Main
"""

### Initialize plot
plt.ion()
fig, axarr = plt.subplots(1, 2)

### Baseline values
raw_signal   = sample(SECONDS, stream_config)
x            = np.linspace(0, SECONDS, len(raw_signal))
fft_signal   = process_fft(raw_signal, fft_config)
fft_peak     = find_peak(fft_signal)
closest_note = find_closest_note(fft_peak)

### Plotting
raw_plot,  = axarr[0].plot(x, raw_signal, linewidth=plot_config["RAW_LINEWIDTH"])
fft_plot,  = axarr[1].plot(fft_signal, linewidth=plot_config["FFT_LINEWIDTH"])
peak_plot, = axarr[1].plot(fft_peak, color="black", linewidth=plot_config["PEAK_LINEWIDTH"])

### Plot text
text_freq    = axarr[1].text(plot_config["X_TEXT_FREQ"], plot_config["Y_TEXT_FREQ"],  "")
text_closest = axarr[1].text(plot_config["X_TEXT_CLOSEST"], plot_config["Y_TEXT_CLOSEST"],  "")
text_note    = axarr[1].text(plot_config["X_TEXT_NOTE"], plot_config["Y_TEXT_NOTE"],  "")

### Raw plot settings
axarr[0].set_ylim(-1000, 1000)
axarr[0].set_xlabel("Seconds")
axarr[0].set_title("Raw signal")

### Fourier plot settings
axarr[1].set_xlim(1, 1000)
axarr[1].set_ylim(0, plot_config["Y_LIM_MAX"])
axarr[1].set_xscale("log")
axarr[1].set_xlabel("Pitch")
axarr[1].set_title("Frequency signal")

### Quick updates were hard to implement.  Thank you to:
### https://stackoverflow.com/a/4098938/9104642
### Updating text needed help from:
### https://stackoverflow.com/a/39228262/9104642
last_peak = fft_peak
while True:
    ### Recalculate
    new_raw  = sample(SECONDS, stream_config)
    new_fft  = process_fft(new_raw, fft_config)
    new_peak = find_peak(new_fft)

    ### Use last peak if signal died off
    if new_peak != 0:
        last_peak = new_peak

    ### Get information for messages
    new_closest  = find_closest_note(last_peak)
    new_note     = determine_note(new_closest)
    new_relative = determine_relative_pitch(new_closest, new_peak)

    ### Redraw
    raw_plot.set_ydata(new_raw)
    peak_plot.set_data([last_peak, last_peak], [0, 1])
    text_freq.set_text(freq_message % new_peak)
    text_closest.set_text(closest_message % str(new_closest))
    text_note.set_text(new_relative % new_note)
    fft_plot.set_ydata(new_fft)

    ### Refresh
    fig.canvas.draw()
    fig.canvas.flush_events()

