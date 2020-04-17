import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import time


class rm3(object):
    """
    Class to make and play random matrices as sound.

    This class takes a random square matrix and raises to an integer power
    so that it generates a pattern that doesn't change when raised to a
    higher power. This matrix is then shifted so its values are frequencies
    in the hearing range, and these frequencies are used to generate a
    waveform that can be played.

    Parameters
    ----------
    central_freq : float, optional
        Central frequency of the melody. Will not necessarily be
        present in the final melody.
    freq_spread : str, array_like, optional
        Indicates the range between the lowest and highest note
        possible in the melody. The options are 'tight' (default),
        'wide', or list/array like [low_freq, high_freq].
    tempo : float, optional
        Tempo of the melody, 100 BPM by default. Can be changed
        after creating the melody.
    dimension : int, optional
        Dimension of the matrix used to generate random melody.
        The matrix will be of shape dimension by dimension.
    min_freq : float, optional
        Minimum frequency allowed in melody. By default it is
        20.0 Hz as a safety measure.
    max_freq : float, optional
        Maximum frequency allowed in melody. By default it is
        20.0 kHz as a safety measure.
    beat_division : int, optional
        How each beat is divided. Used to calculate the length of
        each tone.
    attenuation : float, optional
        Attenuation of waveform to prevent it being too loud.
        Should be a value between 0 and 1.
    clip_level : float, optional
        Clipping applied to final waveform to make it sound a
        little better. Should be a value below that of `attenuation`.
    samples_per_second : int, optional
        Sample rate of waveform.

    Methods
    -------
    make_matrix(n_times=10, show=True)
        Makes a new matrix and shows it.
    show_matrix()
        Shows the current matrix, its values and visualization.
    play(n_repeats=1, loop=False, show=False)
        Plays the current matrix, repeats it 1 time, doesn't
        loop it, doesn't show it.
    stop()
        Stops a currently playing melody.
    make_play(n_repeats=1, loop=False, show=False)
        Makes a new matrix and playes it, repeats it once, doesn't
        loop it, doesn't show it.
    to_notes()
        Shifts frequencies in current matrix to those of closest
        notes.
    to_freqs()
        Returns to raw frequencies. Use only after applying the
        .to_notes() method.
    melody()
        Prints out the melody if method .to_notes() has been run.
    save(filename=None, repeats=1)
        Saves the waveform, repeated once, to a WAVE file. If
        filename is None it will name it with the current date
        and time.
    """

    def __init__(self,
                 central_freq=440.0,
                 freq_spread='tight',
                 tempo=100,
                 dimension=4,
                 min_freq=20.0,  # lower safety stop
                 max_freq=20000.0,  # higher safety stop
                 beat_division=4,
                 attenuation=0.8,
                 clip_level=0.5,
                 samples_per_second=44100
                 ):
        self.dimension = dimension
        self.tempo = tempo
        self.central_freq = central_freq
        self.freq_spread = freq_spread
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.beat_division = beat_division
        self.attenuation = attenuation
        self.clip_level = clip_level
        self.samples_per_second = samples_per_second
        self.waveform = None
        self.rm = None
        self._freq_array = None
        self._melody = None
        self._melody_array = None
        self._melody_notes = None
        self._melody_indices = None

    def make_matrix(self, n_times=10, show=True):
        """
        Makes a square matrix of size dimXdim, raises it to the n_times power, and
        divides by the maximum value. Then prints it and paints it.
        """
        self.rm = np.linalg.matrix_power(np.random.random(
            (self.dimension, self.dimension)), n_times)
        self.rm /= self.rm.max()
        self._freq_array = self._make_freq_array()
        self._reset_melody()
        if show:
            self.show_matrix()

    def _reset_melody(self):
        """Resets melody arrays to None"""
        self._melody = None
        self._melody_array = None
        self._melody_notes = None
        self._melody_indices = None

    def show_matrix(self):
        """Prints and paints matrix rm."""
        fig, ax = plt.subplots()
        ax.imshow(self._freq_array, cmap=plt.cm.inferno)
        # The pattern will be very similar, rounded or not
        print(self._freq_array)
        _ = ax.axis('off')
        fig.show()

    def play(self, n_repeats=1, loop=False, show=False):
        if np.all(self.rm):
            self._generate_waveform()
            if show:
                self.show_matrix()
            sd.play(np.tile(self.waveform, n_repeats),
                    self.samples_per_second, loop=loop)
        else:
            self.make_matrix(self.dimension, show=False)
            self.play()

    def stop(self):
        sd.stop()

    def make_play(self, n_repeats=1, loop=False, show=False):
        """Makes a new matrix and plays it."""
        self.make_matrix(show=show)
        self.play(n_repeats, loop)

    def _make_freq_array(self):
        if self.freq_spread == 'tight':
            freq_range = self.central_freq * 1.5 - self.central_freq * 3 / 4
        elif self.freq_spread == 'wide':
            freq_range = self.central_freq * 2 - self.central_freq / 2
        elif (len(self.freq_spread) == 2) and (self.freq_spread[0] < self.freq_spread[1]):
            freq_range = self.freq_spread[1] - self.freq_spread[0]
        else:
            raise Exception(
                "freq_spread must be 'tight'(default), 'wide', or a list/array/tuple with lower frequency and higher frequency")
        return np.clip((self.rm - self.rm.mean()) * freq_range * 0.5 / self.rm.std() + self.central_freq, self.min_freq, self.max_freq)

    def _generate_waveform(self):
        tone_dur = 60 / (self.tempo * self.beat_division)  # seconds
        each_tone_sample = np.arange(tone_dur * self.samples_per_second)
        if self._melody_array is None:
            self._freq_array = self._make_freq_array()
        else:
            self._freq_array = self._melody_array.copy()
        tone_array = self._freq_array.T * \
            each_tone_sample.reshape(
                each_tone_sample.shape[0], 1, 1)
        tone_array = tone_array.flatten('F')
        # Envelope to avoid clicking
        fraction = 1 / 10  # fraction of tone waveform to be faded in/out
        n_tones = self.rm.flatten().shape[0]  # number of tones
        each_len = each_tone_sample.shape[0]
        ones_len = each_len - 2 * int(each_len * fraction)
        fade_len = int(each_len * fraction)
        # using quadratic fading
        atten_array = np.concatenate((
            (fade_len**2 - (np.arange(fade_len, 0, -1) - 1)**2) / fade_len**2,
            np.ones(ones_len),
            (fade_len**2 - np.arange(fade_len)**2) / fade_len**2
        ))
        self.waveform = np.clip(np.sin(2 * np.pi * tone_array / self.samples_per_second) * np.tile(atten_array, n_tones) * self.attenuation,
                                -self.clip_level,
                                self.clip_level)

    def to_notes(self):
        """
        Rounds frequencies to closest notes.
        """
        if np.all(self.rm):
            notes = np.array(['C', 'C#', 'D', 'D#', 'E', 'F',
                              'F#', 'G', 'G#', 'A', 'A#', 'B'])
            notes_freqs = np.array([
                16.352,  17.324, 18.354, 19.445, 20.602, 21.827, 23.125,   24.5, 25.957,  27.5, 29.135, 30.868,
                32.703,  34.648, 36.708, 38.891, 41.203, 43.654, 46.249, 48.999, 51.913,   55.,  58.27, 61.735,
                65.406,  69.296, 73.416, 77.782, 82.407, 87.307, 92.499, 97.999, 103.83,  110., 116.54, 123.47,
                130.81,  138.59, 146.83, 155.56, 164.81, 174.61, 185.00,   196., 207.65,  220., 233.08, 246.94,
                261.63,  277.18, 293.66, 311.13, 329.63, 349.23, 369.99,   392.,  415.3,  440., 466.16, 493.88,
                523.25,  554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61,  880., 932.33, 987.77,
                1046.5,  1108.7, 1174.7, 1244.5, 1318.5, 1396.9, 1480.0, 1568.0, 1661.2, 1760., 1864.7, 1975.5,
                2093.0,  2217.5, 2349.3, 2489.0, 2637.0, 2793.8, 2960.0, 3136.0, 3322.4, 3520., 3729.3, 3951.1])
            residuals = np.subtract.outer(
                self._freq_array.flatten(),
                notes_freqs)
            notes_indices = np.argmin(abs(residuals), axis=1)
            self._melody_array = notes_freqs[notes_indices].reshape(
                (self.dimension, self.dimension))
            self._melody_notes = notes[notes_indices % len(notes)]
            self._melody_indices = notes_indices // len(notes)
            self.melody()
        else:
            raise Exception('Must first make a Random Matrix')

    def to_freqs(self):
        """Returns a melody that has been rounded to notes in the piano to its original state"""
        if np.all(self._melody_array):
            self._melody_array = None
            self._melody_notes = None
            self._melody_indices = None
            self._melody = None
        else:
            raise Exception(
                'Must first make a Random Matrix and run the .to_notes() method')

    def melody(self):
        """Prints out the melody if method .to_notes() has been run."""
        if np.all(self._melody_array):
            self._melody = ''
            for note, octave in list(zip(self._melody_notes, self._melody_indices)):
                self._melody += note + str(octave) + ' '
            print(self._melody)
        else:
            raise Exception(
                'Must first make a Random Matrix and run the .to_notes() method')

    def save(self, filename=None, repeats=1):
        """
        Saves waveform as a 16-bit .wav file.
        Be sure to use filename that ends in .wav.
        """
        if filename is None:
            filename = time.strftime("%Y%m%d%H%M%S") + '.wav'
        if filename[-4:] != '.wav':
            filename += '.wav'
        sf.write(filename, np.tile(self.waveform, repeats),
                 self.samples_per_second, 'PCM_16')
