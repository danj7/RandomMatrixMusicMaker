# Random Matrix Music Maker (RM^3)

Code to create music from processed random matrices.

## What's it about?

Random matrices in this case are just matrices produced using NumPy's random number generator. If one were to make an image out of a random matrix it would just look like noise. In fact, let's look at this 4*x*4 random matrix as an array and as an image, where darker squares indicate values closer to zero and lighter ones closer to one:

```
[[0.68192887 0.63473388 0.24628448 0.64095272]
 [0.17064282 0.39104756 0.42363141 0.65449383]
 [0.5966333  0.67495844 0.67044778 0.16399837]
 [0.07576034 0.14667625 0.91825815 0.65656918]]
```

<img src="https://user-images.githubusercontent.com/13749006/77688552-a6979f00-6f76-11ea-9714-e585a0063769.png" height="250" title="4x4 Random Matrix (grayscale)">



The image will look the same if made from the original matrix or from a *shifted* matrix, one that's been divided by the maximum value in it. If we take either the original or the *shifted* matrix and raise it to increasing powers we will notice that, after a certain power, the matrix values don't change that much, which means we have a sort of "stable" configuration at some point. We can see this here (I'm using the *inferno* colormap to see more details in the values):

<img src="https://user-images.githubusercontent.com/13749006/77688641-cf1f9900-6f76-11ea-8231-56a2597c91c1.png" height="300" title="Random matrix 4x4 to the 2nd power">  <img src="https://user-images.githubusercontent.com/13749006/77688690-e2caff80-6f76-11ea-854b-b8b18e40d0df.png" height="300" title="Random matrix 4x4 to the 3rd power">  <img src="https://user-images.githubusercontent.com/13749006/77688752-f8402980-6f76-11ea-9f4c-08614d5a6a26.png" height="300" title="Random matrix 4x4 to the 4-th power">  <img src="https://user-images.githubusercontent.com/13749006/77688790-01c99180-6f77-11ea-9213-759b9bf5abb1.png" height="300" title="Random matrix 4x4 to the 10-th power">  <img src="https://user-images.githubusercontent.com/13749006/77688830-127a0780-6f77-11ea-838a-9d646f298da1.png" height="300" title="Random matrix 4x4 to the 25-th power">


After raising it to the 4th power the matrix doesn't change much and, again, darker squares are closer to zero while lighter squares are to one. The grid now has some sections that repeat: Row 1 is similar to Row 3, and Column 1 has all low values. That repetition and "*pattern*" made me think of a tone pattern, like one would find on a [16-step sequencer](https://www.youtube.com/watch?v=BVHJWTX_gIo). So the natural progression of events means I would scale them to frequencies and find a way to produce audio tones from random matrices!


## Requirements and Example

For this to work we need NumPy and Matplotlib, as well as to have installed [`python-sounddevice`](https://python-sounddevice.readthedocs.io), which can be installed using the `conda` package manager or with PyPI:

```
conda install -c conda-forge python-sounddevice
# OR
python3 -m pip install sounddevice
```

The file `rm3.py` has the class to play notes from random matrices so just copy it where you can import it. By default it plays at 100 BPM, with a maximum frequency of 440 Hz.

```python
>>> from rm3 import rm3
>>> wave = rm3()
>>> wave.play() #a random matrix is created and played
>>> wave.show() #shows current matrix
>>> wave.make_matrix() #makes and processes a new matrix, and shows it
>>> wave.tempo = 180 #now we want something fast
>>> wave.play(loop=True) #loops it indefinitely
>>> wave.stop()
```

So far this is just playing frequencies with relation to actual notes. However, you can shift these frequencies to the closest notes and displays them.
```python
>>> wave = rm3(tempo=75, max_freq=220)
>>> wave.to_notes()
A#5 A#5 G5 B5 G#6 G6 F6 A6 D#6 D6 C6 E6 B5 B5 G#5 C6
>>> wave.play(show=True) #play and show matrix
```

## To do
* save data file and/or waveform
* maybe some effects or FM synth?
* multiple streams/sounds at the same time?
