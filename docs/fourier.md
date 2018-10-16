# Fourier Transforms
## DFT
The dft_slow function is a plain implementation of discrete Fourier transformation. 

The fft_vectorized function depends on this function.

Parameters:
| Parameter | Description |
| --- | --- |
| input_data | Array containing the values to be transformed. |

Returns an array containing the transformed values.

Example usage: 
```python
>>> from Asteria import fourier
>>> fourier.dft_slow([1,2,3,4])
array([10.+0.00000000e+00j, -2.+2.00000000e+00j, -2.-9.79717439e-16j, -2.-2.00000000e+00j])
```

## FFT
The fft_vectorized function is a vectorized, non-recursive version of the Cooley-Tukey FFT

Gives the same result as dft_slow but is many times faster. 

Parameters:
| Parameter | Description |
| --- | --- |
| input_data | Array containing the values to be transformed. |

Returns an array containing the transformed values.

Example usage: 
```python
>>> from Asteria import fourier
>>> fourier.fft_vectorized([1,2,3,4])
array([10.+0.00000000e+00j, -2.+2.00000000e+00j, -2.-9.79717439e-16j, -2.-2.00000000e+00j])
```