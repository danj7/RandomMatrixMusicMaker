import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


class rm3(object):
    """
    Class to make and play random matrices as sound.
    TO DO:
    * implement FM synth with detuning
    * saving array and waveform
    """

    def __init__(self,
                 dimension=4,
                 tempo=100,
                 max_freq=440.0,
                 min_freq=20.0,
                 beat_division=4,
                 attenuation=0.8,
                 clip_level=0.5,
                 samples_per_second=44100
                 ):
        self.dimension = dimension
        self.tempo = tempo
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.beat_division = beat_division
        self.attenuation = attenuation
        self.clip_level = clip_level
        self.samples_per_second = samples_per_second
        self.waveform = None
        self.rm = None
        # self._rm_bare = None
        self._rm_backup = None
        self._melody = None
        self._melody_notes = None
        self._melody_indices = None

    def make_matrix(self, n_times=7, show=True):
        """
        Makes a square matrix of size dimXdim, raises it to the n_times power, and
        divides by the maximum value. Then prints it and paints it.
        """
        self.rm = np.linalg.matrix_power(np.random.random(
            (self.dimension, self.dimension)), n_times)
        self.rm /= self.rm.max()
        if show:
            self.show_matrix()

    def show_matrix(self):
        """Prints and paints matrix rm."""
        fig, ax = plt.subplots()
        ax.imshow(self.rm, cmap=plt.cm.inferno)
        # The pattern will be very similar, rounded or not
        print(self.rm * self.max_freq)
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

    def make_play(self, n_repeats=1, loop=False):
        self.make_matrix(show=False)
        self.play(n_repeats, loop)

    def _generate_waveform(self):
        tone_dur = 60 / (self.tempo * self.beat_division)  # seconds
        each_tone_sample = np.arange(tone_dur * self.samples_per_second)
        tone_array = self.rm.T * \
            each_tone_sample.reshape(
                each_tone_sample.shape[0], 1, 1)
        if self._rm_backup is None:
            tone_array *= self.max_freq  # its not been rounded
        tone_array = np.clip(tone_array.flatten('F'), self.min_freq, None)
        # Envelope to avoid clicking
        fraction = 1 / 10  # fraction of tone waveform to be faded in/out
        n_tones = self.rm.flatten().shape[0]  # number of tones
        each_len = each_tone_sample.shape[0]
        ones_len = each_len - 2 * int(each_len * fraction)
        fade_len = int(each_len * fraction)  # using quadratic fading
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
                16.352,   17.324,   18.354,   19.445,   20.602,   21.827,
                23.125,   24.5,   25.957,   27.5,   29.135,   30.868,
                32.703,   34.648,   36.708,   38.891,   41.203,   43.654,
                46.249,   48.999,   51.913,   55.,   58.27,   61.735,
                65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
                92.499,   97.999,  103.83,  110.,  116.54,  123.47,
                130.81,  138.59,  146.83,  155.56,  164.81,  174.61,
                185.,  196.,  207.65,  220.,  233.08,  246.94,
                261.63,  277.18,  293.66,  311.13,  329.63,  349.23,
                369.99,  392.,  415.3,  440.,  466.16,  493.88,
                523.25,  554.37,  587.33,  622.25,  659.26,  698.46,
                739.99,  783.99,  830.61,  880.,  932.33,  987.77,
                1046.5, 1108.7, 1174.7, 1244.5, 1318.5, 1396.9,
                1480., 1568., 1661.2, 1760., 1864.7, 1975.5,
                2093., 2217.5, 2349.3, 2489., 2637., 2793.8,
                2960., 3136., 3322.4, 3520., 3729.3, 3951.1])
            residuals = np.subtract.outer(
                self.rm.flatten() * self.max_freq, notes_freqs)
            notes_indices = np.argmin(abs(residuals), axis=1)
            self._rm_backup = self.rm.copy()
            self.rm = notes_freqs[notes_indices].reshape(
                (self.dimension, self.dimension))
            self._melody_notes = notes[notes_indices % len(notes)]
            self._melody_indices = notes_indices // len(notes)
            self.melody()
        else:
            raise Exception('Must first make a Random Matrix')

    def to_freqs(self):
        """Returns a melody that has been rounded to notes in the piano to its original state"""
        if np.all(self._rm_backup):
            self.rm = self._rm_backup.copy()
            self._rm_backup = None
            self._melody = None
        else:
            raise Exception(
                'Must first make a Random Matrix and run the .to_notes() method')

    def melody(self):
        """Prints out the melody if method .to_notes() has been run."""
        if np.all(self._rm_backup):
            self._melody = ''
            for note, octave in list(zip(self._melody_notes, self._melody_indices)):
                self._melody += note + str(octave) + ' '
            print(self._melody)
        else:
            raise Exception(
                'Must first make a Random Matrix and run the .to_notes() method')
