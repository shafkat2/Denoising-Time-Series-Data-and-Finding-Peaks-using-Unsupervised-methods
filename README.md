# Denoising-Time-Series-Data-and-Finding-Peaks-using-Unsupervised-methods
A dataset, collected under an industrial setting, often contains a significant portion of noises. In many cases, using trivial filters is not enough to retrieve useful information i.e., accurate value without the noise. One such data is time-series sensor readings collected from moving vehicles containing fuel information. Due to the noisy dynamics and mobile environment, the sensor readings can be very noisy. Denoising such a dataset is a prerequisite for any useful application. It has led us to develop a system that can remove noise and keep the original value and help vehicle industry, fuel station, and power-plant station that require fuel. In this work, we have only considered the value of fuel level, and we have come up with a unique solution to filter out the noise of high magnitudes using several algorithms such as interpolation, extrapolation, spectral clustering, agglomerative clustering, wavelet analysis, and median filtering. We have also employed peak detection and peak validation algorithms to detect fuel refill and consumption in charge-discharge cycles. We have used the R-squared metric to evaluate our model, and it is 98%. In most cases, the difference between detected value and real value remains within the range of ±1L.
