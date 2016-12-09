# mvts: Minimum Variability Time Scale for light curves

This is a simple package which can be used to plot the wavelet spectrum, 
based on the continuous wavelet transform.

## Install

```
> git clone https://github.com/giacomov/mvts.git
> cd mvts
> python setup.py install
```

(you might need to become root, or alternatively, to add the '--user' option 
to the last command)

## Usage

```python
import mvts

results, fig = mvts.wavelet_spectrum()

```

The ```results``` dictionary will contain the results of the wavelet transform,
while the figure will contain the wavelet spectrum.