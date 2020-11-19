**Analysis and Denoising of Time Series Data Using Unsupervised
Methods**

> Shafkat Waheed1+, B. M. Raihanul Haque1+, Abir Roy1, Mst. Shapna
> Akter1, Sumit Ranjan Chakraborty2, Shakran Hayat2, and M.R.C.
> Mahdy1,2\*
>
> *1 Department of Electrical & Computer Engineering, North South
> University, Bashundhara, Dhaka, 1229, Bangladesh.*
>
> *2 Pi Labs Bangladesh Ltd., Eden Center, 2/1/E, Toyenbee Rd, Dhaka
> 1000, Bangladesh.*
>
> \+ Equal contribution

\*Corresponding Author: <mahdy.chowdhury@northsouth.edu>

> **ABSTRACT:**
>
> **A dataset, collected under an industrial setting, often contains a
> significant portion of noises. In many cases, using trivial filters is
> not enough to retrieve useful information i.e., accurate value without
> the noise. One such data is time-series sensor readings collected from
> moving vehicles containing fuel information. Due to the noisy dynamics
> and mobile environment, the sensor readings can be very noisy.
> Denoising such a dataset is a prerequisite for any useful application.
> It has led us to develop a system that can remove noise and keep the
> original value and help vehicle industry, fuel station, and
> power-plant station that require fuel. In this work, we have only
> considered the value of fuel level, and we have come up with a unique
> solution to filter out the noise of high magnitudes using several
> algorithms such as interpolation, extrapolation, spectral clustering,
> agglomerative clustering, wavelet analysis, and median filtering. We
> have also employed peak detection and peak validation algorithms to
> detect fuel refill and consumption in charge-discharge cycles. We have
> used the R-squared metric to evaluate our model, and it is 98%. In
> most cases, the difference between detected value and real value
> remains within the range of ±1L.**

**Keywords: Interpolation, Extrapolation, Spectral clustering,
Agglomerative clustering, Wavelet analysis, Median filtering.**

**\
**

1.  **Introduction**

> Data denoising or removing noise from a dataset is a challenging and
> fascinating topic for the researchers. Each dataset is different in
> terms of data distribution and noise. Dataset might be a set of images
> or audio signals or any other generic data. These noises are generated
> due to device malfunctioning, human limitation, machine limitation,
> improper approaches of collecting and preserving data, etc.
>
> However, in this work, the focus revolves around the industrial
> dataset that has issues like the severely noisy environment, and this
> dataset contains fuel level information found in a fuel tank. Whether
> this is a vehicle industry or fuel station or a power plant, which is
> being operated using fuel, they have almost similar problems. Since
> this is a time-series data, any given point within a certain time
> frame either indicates charging or discharging, and noise present in
> the dataset gives a false representation of peak or consumption. Here,
> false representation stands for noisy data that plagues the actual
> value. For instance, at any given time, the value obtained from
> sensors could be 40 liters, whereas the actual value could be of 60 or
> 30 liters. Using a typical filter while cleaning the dataset, causes
> data loss, and fails to remove noise properly. In an attempt to reduce
> noise from time-series data representing the fuel volume of a vehicle;
> some related works have been discussed below.
>
> So far, most of the researches has been conducted on medical images
> \[1-5\]. In all these papers, the dataset comprises MRI images, X-ray
> images, CT scan images, PET scan images, and so on. Some of the
> proposed methods incorporate NN (neural network), SVM (support vector
> machine), denoising autoencoder, etc. The problem with these methods
> is that: in the case of a complete unsupervised scenario, all the
> proposed models, especially NN, perform poorly. In our case,
> implementing a device to a car to get accurate real-time values has
> not been feasible. Therefore, NN has not been an option to deduce real
> value. For similar reasons, the use of LSTM (long short term memory),
> GAN (generative adversarial network), RNN (recurrent neural network),
> and other classifiers, which have been proposed by some authors
> \[6-8\], have not been possible either.
>
> Another notable area where denoising techniques have been implemented
> is audio signals \[9-12\]. Audio signals usually come from speech,
> background noises, or any other automated source (electronic,
> electromagnetic, acoustic). The suggested methodology is heavily
> dependent on filters such as Kalman filter, Butter Worth, Chebyshev,
> Elliptical Filters, and so on. In the medical sector, ECG
> (electrocardiogram) signals are also cleaned using filters such as low
> pass filter \[13\]. Apart from these, deep learning methodologies
> \[9\] and non-linear diffusion \[12\] have also been considered. We
> have already ruled out a deep learning technique due to the
> unsupervised nature of the data, which we have dealt in this work. The
> other problem is that: relying just on filters produces poor results
> due to high regularization. In contrast, our model tends to be
> generic, which means it will work efficiently on any vehicle with
> ambiguous noise patterns.
>
> It is evident that: the contemporary techniques are focused on
> denoising images or audio signals. Although denoising techniques are
> applied in the realm of IoT (internet of things) by using ANN
> (artificial neural network) \[14\], ToF (Time-of-Flight) data by using
> GAN \[15\], and vibration sensor data by using TFM (time-frequency
> manifold) \[16\], none of the solutions seems suited for our dataset
> due to unsupervised nature of the data. In our case, we had to tackle
> two parallel problems; one is to minimize data loss, and another is to
> get the actual value in a completely unsupervised manner.
>
> Among the few works in literature, Manmohan et al. \[17\] have
> thoroughly studied and implemented the process of wavelet analysis.
> They have also proposed a method which includes the median filter
> \[17\]. They have suggested that: ordinary median filter indicates an
> improvement against other filtering techniques i.e., the mean filter.
> Also, two thresholding methods and wavelet allow them to retrieve
> original features of the image effectively. Additionally, implementing
> median filtering is suggested in reference \[18\]. This inspires us to
> implement wavelet analysis and the median filter in our dataset.
>
> Among many other techniques, clustering has been implemented in
> amplicon sequencing by Kaisa Koskinen et al. \[19\]. The impact of
> clustering is huge on the number of operational taxonomic units
> (OTUs). Priyam Chatterjee et al. have suggested the positive impact of
> clustering in data denoising as well \[20\]. Therefore, clustering has
> been considered as an imperative option for our dataset. For time
> series and other types of data, such as for predicting time-averaged
> force \[21\], forecasting price of the market \[22-23\], or classify
> visual pollutants from an image \[24\], machine learning and deep
> learning can also serve as highly effective tools. But for our case,
> due to the aforementioned reasons, a different approach has been
> implemented.
>
> To test the efficiency of our model, we have been provided with a
> specific industrial dataset. This dataset contains the
> charging-discharging cycle value of the moving vehicle. All the
> vehicles have been operative on the street in Bangladesh. Due to the
> poor condition of a few streets in Bangladesh, noise may have been
> prominent in the dataset. Since, barely any work has been done in
> denoising time-series data containing fuel information retrieved from
> the industrial domain; therefore, to accomplish this, we have tried
> many methodologies to find out the optimal solution. Our system has
> started with simple interpolation, extrapolation, and white noise
> removal techniques. Here, by white noise, we are referring to random
> noise present in a dataset whose mean value is zero and has a finite
> variance \[25\]. Then two clustering methods \[26-27\] have been
> brought to isolate noise from real value. Wavelet analysis \[28-30\]
> has cleaned it further through wavelet transformation and inverse
> transformation. Finally, median filtering \[31\] has been used to give
> the final push. Furthermore, many custom algorithms have been used to
> arrange previously mentioned algorithms and compare different datasets
> obtained from those algorithms to narrow down the refill value.
>
> Apart from the unorthodox noises caused by the reason mentioned above,
> there is also an unusual noise present in the dataset. Usually, when
> the engine or machine is switched off, there is no charging or
> discharging occurring within a certain time period. This yields a
> stationary or constant set of values within that time frame. Our
> system has required to deal with this issue, and for most of the
> parts, it has accomplished successfully.
>
> At first, we have developed our model using a concatenated dataset
> containing information from multiple vehicles. Then, to test it
> further we have been provided with an additional dataset of various
> vehicles. The initial dataset contains over 100,000 data points, which
> have been narrowed down to 37 peak values. To evaluate our result
> properly, we have used the R-squared metric and compared our result
> with previous work that has been done on similar datasets \[32-38\].
> The R-squared value of the final result is 0.98, and RMSE is 1.49. Our
> system detects not only the refill of fuel but also the consumption
> rate while minimizing data loss in a very noisy industrial
> environment.

2.  **Methodology**

> All the data have been collected from Pi Labs Bangladesh Ltd.
> Initially, from 9 vehicles, roughly 1,00,000 data points have been
> collected and used for our system. Data flow is shown in Fig.1, which
> demonstrates the cycle starting from raw data to processed data. It
> begins with employing industrial settings in a moving vehicle. Then
> some generic transmission occurs. Our algorithm can reduce noise from
> the time-series data while it is streamed from source to destination
> through a server.

![A close up of text on a white background Description automatically
generated](media/image1.png){width="6.5in" height="4.229166666666667in"}

Fig.1 A general overview of the data cycle. The industrial setting is
implanted in a vehicle, and it collects and sends data to the server.
Inside the server, our algorithms are implemented where the data
traverse through each phase. After successful peak detection, the data
is sent to the mobile application (beta version) from where the result
can be displayed.

> The scope of this project revolves around the algorithms. We wanted to
> test whether real-time data can be displayed, and this is why a mobile
> application prototype has been built. An enhanced overview of our
> algorithm is represented in Diagram.1. Our system begins cleaning the
> dataset with simple extrapolation and interpolation. This removes some
> trivial types of noises i.e., white noise. Then two sets of clustering
> methodologies (spectral and agglomerative) isolate noise from the
> actual value. After that, the dataset is stored and sent to the next
> phase to perform wavelet analysis. This generates two datasets to be
> compared and keep the common values. A similar strategy is undertaken
> for median filtering and wavelet analysis. Finally, when all prior
> operation is done, we have performed a final validation to get the
> peak values and consumption rate.

![A screenshot of a cell phone Description automatically
generated](media/image2.png){width="3.827777777777778in"
height="8.083333333333334in"}

Diagram.1 After receiving the data, it has been sent for preprocessing,
where we have used a few data wrangling techniques to deal with missing
values and white noises. This data is then stored, and data is sent to
the next phase for wavelet analysis. These two sets of data are being
compared to eliminate redundant data points. A similar process can be
observed for median filtering and wavelet transformation. Lastly, a
final comparison is made to detect peaks.

> Initially, the dataset is comprised of a massive amount of noise. As a
> result, it has been hard to isolate them from the actual value just
> through observation. Fig.2 (a) gives a generic outlook of the data
> set.

![A close up of text on a black background Description automatically
generated](media/image3.png){width="6.5in" height="7.291666666666667in"}

Fig.2 The outlook of our dataset after each step from initial to
processed. The congested regions are the indication of data noise.
(a)-(b) Raw data and data after white noise removal, respectively. In
both cases, heavy noise is present in the dataset. (c)-(g) Graphical
representation of data after clustering, wavelet analysis (1st), median
filtering (1st), wavelet analysis (2nd), and median filtering (2nd). A
significant portion of the noise is removed in each step. (h) Processed
data, containing 37 peaks.

> In this figure, from right to left, we see downward spikes. This slow
> decrease indicates the consumption rate over a certain time period.
> The sharp jump from the spike represents refuel. Each segment in this
> graph illustrates the individual vehicle\'s fuel level.

1.  **Data characteristics**

> Before designing a model, it is important to know some characteristics
> of the data so that similar work can be replicated, or the designed
> module can be implemented on a similar data set. The goal was to plot
> a histogram within a certain frequency. As seen in Fig.3, there are
> multiple intervals and corresponding frequency where data points are
> grouped within a certain range.

![A screenshot of a cell phone Description automatically
generated](media/image4.png){width="6.5in"
height="3.0930555555555554in"}

Fig.3 Histogram shows the distribution of the dataset

> To understand the characteristics better, we have calculated the mean
> and median of the data along with the 1st and 3rd quartile range. Our
> objective was to find out whether our median is closed to the 1st or
> 3rd quartile range. Our mean = 48.327, standard deviation = 12.008,
> median = 52.316, 1st quartile range = 40.790 and 3rd quartile range =
> 56.837. Obviously, the median is close to the 3rd quartile range, and
> the mean is less than the median. We have calculated the moments of
> the dataset to identify the distribution. But we could not find any
> specific distribution related to the dataset, which is why we have
> undertaken an unsupervised approach to denoise the dataset.

2.  **Data Preprocessing**

Initially, some rows contained zero values. This could contaminate the
results of further processes, and therefore data needed to be tackled in
an efficient manner where data replacement conforms to the pattern of
non-zero values. Preprocessing involves white noise removal using
various data wrangling techniques and applying
interpolation-extrapolation to recover the lost points. Also, we have
differentiated the dataset to create a new feature.

Interpolation is a method that can be achieved using discrete data set
and by forming general formula within a certain range. In our case,
linear interpolation was used, which can be expressed as
$\ y = y_{1} + \left( \frac{y_{2} - y_{1}}{x_{2} - x_{1}} \right)(x - x_{1})$.
Here, unknown points are denoted as y for a certain value of x.

Extrapolation operates outside of the range of the observation. It is
subjected to higher uncertainty, as opposed to interpolation, since it
fills out data based on the relation of other data points. In this
research, midpoint extrapolation was incorporated, denoted as
$\left( x,y \right) = (\frac{x_{1} + x_{2}}{2},\frac{y_{1} + y_{2}}{2})$,
and which is operative for one unknown point.

> Interpolating and extrapolating data points were achieved through the
> following process. At the beginning, it is checked whether the data
> points contain 0 values or not. If it does so, all those values are
> set to NULL. Finally, through interpolation and extrapolation, new
> data has been generated and is being stored.
>
> Apart from that, a general form of noise also known as random noise,
> has been removed from the data set. The main characteristic of the
> noise includes the equal intensity of the signal at different
> frequencies. If the mean equals zero, then data is considered white
> noise, as data is spread across the negative and positive potion of
> the graph in an equal manner. The white noise is excluded from the
> dataset by using the following formula iteratively:
> $\alpha = (x_{i} - x_{i + 1}) + (x_{i} - x_{i - 1})$. However, there
> is a strict condition for white noise removal, which is,
> $x_{i} \neq x_{i + 1}$ and$\ x_{i} \neq x_{i - 1}$. The iterative
> process can be represented as $y_{i} = x_{i}\alpha$. After successful
> completion of preprocessing, we have gotten a visual representation of
> the data set, as presented in Fig.2 (b).

3.  **Clustering**

> Clustering is an unsupervised learning method. It is used to group
> data according to their respective characteristics. Any data in a
> single cluster possesses a similar characteristic as every other data
> point in the same cluster. Right after the preprocessing is done, two
> clustering techniques have been incorporated. These clustering methods
> work together to separate noisy data from correct data.
>
> Spectral clustering is best suited when the variation of data is not
> much. It clusters the data based on the density of data points. The
> state of being close proximity to each other is called affinity, and
> this phenomenon can be described by the affinity matrix. Different
> vectors from this matrix can be extracted using principal component
> analysis (PCA), which later leads Eigenvectors to be formed. These
> vectors are referred to as feature vectors of each object of affinity
> or Laplacian matrix.
>
> Hierarchical clustering, also known as Agglomerative clustering,
> generates a tree where roots are the lowest point of the data-set, and
> leaf nodes consist of values that are greater than the values of the
> root node. It uses the bottom-up approach to group the data points in
> a hierarchical manner. Every data point groups together in its own
> cluster. This goes on for all data points, and these clusters are then
> joined using a greedy approach. The greedy approach involves merging
> two most similar clusters together.
>
> To perform the clustering on the entire dataset we have incorporated
> the following formula,
> $y_{s,h} = \alpha f_{s}\left( x_{i} \right) + (1 - \alpha)f_{h}\left( x_{i} \right)$.
> Here, $f_{s}$ and $f_{h}$ represent the mechanism of spectral and
> agglomerative clustering respectively as functions. At the start of
> clustering, we set a constant value denoted as threshold $T$ whose
> value is 0.1. Then standard deviation is being calculated based on all
> the data points. As agglomerative clustering uses a dendrogram to
> cluster that data it works well on the dataset having higher
> deviational value. As opposed to that, spectral clustering checks on
> the affinity of the data points to group them; That is why it is
> better suited for dataset having lower deviational value. Due to
> variation of standard deviation after each iteration, we used two
> clustering. If $T$ is greater than standard deviation, then
> $\alpha = 0$ and data points are grouped using Agglomerative
> clustering, and if it is less then, $\alpha$ is set to 1, and data are
> grouped together using spectral clustering. Each iterative step is
> incremented by 200. After clustering, many data points get eliminated
> as part of the noise removal technique. Interpolation and
> extrapolation are used to fill out those data points. Fig.2 (c) is a
> visual representation of the effect of clustering on the data set.

4.  **Wavelet Analysis**

> Unlike any other Fourier transformation methodology, wavelet
> transformation is used for transforming data represented in the time
> domain to the frequency domain. In previous work, we have found that
> it has the ability to compress an image efficiently. By managing
> factors like shifting and scaling, it can decompose an image to
> multiple lower resolution images. The waves have features like varying
> frequency, limited duration, and zero average value. This is also
> eligible to remove high-frequency noise from any time-series data with
> non-continuous peaks. The implementation of wavelets revolves around
> implementing two different transformations and incorporating one
> threshold function. These transformations are wavelet transformation
> and inverse wavelet transformation. The wavelet transformation is
> achieved by the following formula,
> $C\left( \tau,\ s \right) = \frac{1}{\sqrt{s}}\int_{t}^{}{f(t)\psi^{*}(\frac{t - \tau}{s})dt}$.

![A picture containing tree, flower Description automatically
generated](media/image5.png){width="4.686111111111111in"
height="8.78125in"}

Diagram-2 An illustration of wavelet analysis containing two
transformations.

> Above is the formula for continuous wavelet transformation where
> $\tau$ and $s$ are transition parameter and scale parameter
> respectively, $\frac{1}{\sqrt{s}}$ is normalization constant, and
> $\int_{t}^{}{f(t)\psi^{*}(\frac{t - \tau}{s})dt}$ is the mother
> wavelet. The inverse operation is carried out by the following
> function:
> $f\left( t \right) = \ \frac{1}{\sqrt{s}}\int_{\tau}^{}{\int_{s}^{}{C\left( \tau,\ s \right)\psi(\frac{t - \tau}{s})d\tau ds}}$.
>
> Discrete wavelet transformation is a bit straight-forward than this
> and, to state the obvious, free from the integral operation. The
> formula for discrete wavelet transformation is
> $a_{\text{jk}} = \sum_{t}^{}{f(t)\psi_{\text{jk}}^{*}(t)}$ and inverse
> discrete wavelet transformation can be written as such,
> $f\left( t \right) = \sum_{k}^{}{\sum_{j}^{}{a_{\text{jk}}\psi_{\text{jk}}(t)}}$.
> These formulas provide simultaneous localization in time and scale,
> sparsity, adaptability and linear time complexity; which allow noise
> filtering, image compression, image fusion, recognition, image
> matching and retrieval efficiently. Finally, an additional threshold
> function can be represented as such to improve the proposed model,
> $threshold = alpha*noise*\sqrt{log(data\_ size)}$ and in this formula,
> alpha is a constant and noise = absolute median value. We incorporated
> the discrete approach in our model.
>
> This reserves the original signal but without the noise. After the
> implementation, the peak value was shifted by 5200 points. Diagram-2
> provides the mechanism and Fig-2 (d) and (f) show the result after
> wavelet analysis.

5.  **Median Filtering**

> Median filtering is a non-linear filtering technique, and it removes
> noise from the data set while preserving the valuable data. In our
> case, the median filter is to traverse through the signal one by one.
> It generates a window to slide through the data entry, and this window
> is generated based on the pattern of the neighboring window. This
> replaces each entry with the median of neighboring entries and what
> remains is the peak. Fig.2 (e) and (f) show results after median
> filtering applied for the first time on the data set. Our data has one
> dimension. A window of the median filter contains the first few
> preceding and following entries. To carry out this task, we used this
> formula,
> $y\left( X_{\text{top}},X_{\text{bottom}} \right) = M\{ x_{i},x_{i + 1}\}$,
> where $M$=median, $X_{\text{top}}$=Initial point,
> $X_{\text{bottom}}$=Last point, and
> $\left\{ x_{i},x_{i + 1} \right\} \in X$.
>
> Once what algorithm to use is sorted out, we have been required to
> write our own custom algorithms where we can use clustering, filtering
> methodologies efficiently. It is necessary to choose data window and
> iterative steps carefully so that we can minimize the data loss. Our
> custom algorithms include one peak detection and three peak validation
> methods.

6.  **Peak detection**

> This is the part where peaks are detected. The algorithm uses the
> moving average to find any sudden changes in the data points. As we
> pass through data points, we calculate the simple moving average with
> $SMA = \frac{\sum_{}^{}x_{i}}{n}$. We used a window of 3 points to
> find any discrepancy in the data. So, for every three adjacent point
> $\text{SMA}_{3} = \frac{\sum_{i = 1}^{n = 3}x_{i}}{3}$. Any deviation
> of more than 4 units is considered a peak, which means if
> $|x_{i} - \text{SMA}_{3}| > 4$, then $x_{i}$ is a valid peak.
>
> There are three peak validation methods in this system, among which
> two are almost identical. These validation methods validate the data
> points, whether they are valid peak points or not. The prior two peak
> validation methods include the following:

7.  **Peak validation (first and second)**

> At the start, peak validation data is sent to two different
> directions. One part is operative using clustering and peak detection
> algorithm. Another part is operative using a combination of
> clustering, wavelet analysis, and peak detection algorithm. This
> produces two different datasets. After that, each entry of one dataset
> is evaluated against the corresponding entry of other data set. If
> there is a similarity, we validate the peaks; otherwise, we discard
> it. This parallel processing gets the work done fast without
> interrupting each other. The formula for peak validation (first) is,
> $f_{\text{rp}} = \left| f_{\text{pcp}} - f_{\text{pwp}} \right| \leq 100$
> where $f_{\text{rp}}$ = valid peak points, $f_{\text{pcp}}$ = data
> points obtained after performing peak detection on clustered data
> points = $SMA(y_{s,h}(x_{i}))$ and $f_{\text{pwp}}$ = data points
> obtained after performing peak detection on data points retrieved from
> implementation of clustering and wavelet analysis ($W_{A}$) =
> $SMA(W_{A}(y_{s,h}\left( x_{i} \right)))$. In order to get valid peak
> $f_{\text{rp}}$ must be less than or equal to 100. It is a comparison
> between values obtained by only after clustering and a combination of
> clustering and wavelet. Real peaks are validated based on similarity.
>
> The second peak validation method, as mentioned earlier, works
> similarly. The only difference is that, in second peak validation,
> median filtering has been used instead of clustering. Likewise, it is
> a comparison between the two datasets. One is obtained after
> performing just filtering, and another is acquired through
> implementing a combined system of filtering and wavelet. So, the
> formula for second peak validation is,
> $f_{\text{rp}} = |f_{\text{pcp}} - f_{\text{pmp}}| \leq 100$ where
> $f_{\text{pmp}}$ = data points obtained after performing peak
> detection on data points retrieved from the implementation of
> clustering and median filtering ($M_{F}$) =
> $SMA(M_{F}(y_{s,h}\left( x_{i} \right)))$. As part of the peak
> validation, the effect of wavelet transformation and median filtering,
> which is implemented for the second time, can be found in Fig.2 (f)
> and Fig.2 (g), respectively.

8.  **Peak validation (final)**

> This peak validation method is operative on the final data set before
> producing the final result. Three consecutive data points, previous =
> i-1, current = i and next = i+1, are considered for data traversal.

![A close up of text on a white background Description automatically
generated](media/image6.png){width="5.134027777777778in"
height="7.125in"}

Diagram-3 Third and final peak validation method. It is a comparison
between three adjacent data points. This exerts the final 37 data
points, which correspond to the peak value.

> We have defined two variables, such as A = previous -- current and B =
> current -- next, to check the distance and difference of A and B.
> Distance refers to the position between A and B, and the difference is
> the value obtained by subtracting A and B. If the distance is less
> than 30 and the difference is less than 5, then the value is removed
> from the peak. This final procedure isolates the peak from the rest of
> the data to have a proper evaluation of peak detection. After this
> peak validation, we obtain 37 peak points. After getting the peaks, we
> can easily calculate the consumption rate by finding the difference
> between peaks. This final procedure isolates the peak from the rest of
> the data to have a proper evaluation of peak detection, as illustrated
> in Diagram-3.

Result and Analysis
-------------------

> Our paper has focused primarily on noise reduction and peak detection
> from the industrial dataset, which is retrieved from vehicles. To
> achieve this target, we have faced an intricate trade-off; detecting
> peaks while minimizing data loss. To get the actual value from the
> noisy time series dataset, we have implemented different algorithms
> mentioned in the previous section. Implementation of similar
> algorithms can be found in reference \[32\], where both wavelet
> transformation and adaptive denoising algorithms are implemented on
> non-linear (chaotic) time-series datasets. The performance of the
> wavelet analysis is poor as it has not been systematically studied for
> chaotic signals, and thus, no conclusive proof is given whether the
> wavelet-based noise reduction techniques truly are the best. Chaotic
> signals usually have a broad-band spectrum that overlaps with the
> spectrum of noise. In our case, after minimizing the noise using
> several algorithms, wavelet transformation is implemented as a
> catalyst to trim the noise further.
>
> In another work \[33\] the authors have run wavelet transformation on
> the traffic volume time series dataset. However, they could not deal
> with past correlated samples due to the longer lags into the traffic
> volume pattern. Likewise, a threshold-based wavelet denoising
> algorithm is applied on hydrological time series data to acquire a
> detailed approximation of rainfall and runoff signals at various
> resolution levels, respectively \[34\]. It has been deduced that there
> are both positive and negative trends in each zone for the monsoon
> datasets, which shows the same periodicity, and thus, it affects the
> accuracy of acquiring approximate and detailed rainfall and runoff
> signals. An almost identical issue has emerged in other work \[35\],
> where the author used discrete wavelet transformation on hydrological
> time series data and found the limitation of positive and negative
> trends. In contrast, our approach has successfully removed all
> possible noises. In reference \[36\], different denoising and
> gap-filling methods such as interpolation, the Singular Spectrum
> Analysis, and the Lomb-Scargle algorithm. They have further suggested
> to select diverse methods based on the problem type of each noise
> pattern of the dataset. Each of the algorithms has its strength and
> drawbacks i.e. the Lomb-Scargle method is sensitive in the presence of
> a strong trend in the raw data. This sensitivity may boost the power
> spectrum at low frequencies which may therefore mask other frequencies
> that would be significant in the absence of that limitation. Due to
> this issue, the method produces false, intermittent peaks in these
> systematic gaps, and the overall experiment shows the poor result. One
> the other hand, in our case, we achieved semi-accurate results by
> mashing up different algorithms according to the type of noises. Our
> two-layered noise reduction method has detected peaks while
> eliminating the noise based on a threshold value.
>
> Other investigations have been done on the approach of peak detection
> in the time series data set. In one such work \[37\], the authors have
> proposed a continuous wavelet transform (CWT)-based pattern matching
> method for detecting strong and weak peaks without performing data
> preprocessing. Though they have efficiently detected the strong and
> weak peaks on the raw data but it can only be applicable for the
> dataset with default parameters, and the computational load is
> comparatively high. The experimental result they have achieved shows
> one isolated false positive identification from the spectrums. The
> work presented in reference \[38\] has shown several statistical ways
> to compute the time-series peak function. They have experimented on a
> dataset consisting of annual sunspot data for years 1700 to 2008,
> where the best result shows almost accurate results but failed to
> detect more than two peaks. Conversely, our algorithm is successfully
> operative in detecting multiple peaks.

![A screenshot of a cell phone Description automatically
generated](media/image7.jpg){width="6.5in" height="5.379861111111111in"}

Table--1 List of detected peaks as compared to the real value. For the
most part, the error is minimum.

> As seen in Table-1, for the most part, our module can detect peaks
> with an error of ±1𝐿. Those values whose error rate is higher than 5
> liters are due to the reason mentioned above. However, the R-squared
> score is 0.98, and RMSE is 1.49, which indicates that our module is
> highly efficient. The graphical representation of cleaned data after
> all noise removal can be seen in Fig.2 (h).
>
> Due to hardware malfunction or some similar reason, at different time
> periods, the obtained data showed bizarre patterns, which has caused
> discrepancy in finding the correct result. In the highway, due to the
> high speed, the fuel consumption rate remains constant, as the device
> has its limitation. As a result, there is a huge data loss during this
> time period. If the dataset collected has no data loss, then the
> result can be significantly improved where the accuracy may exceed
> ours.

![A screenshot of a cell phone Description automatically
generated](media/image8.png){width="6.375890201224847in"
height="3.6463418635170606in"}

Fig.4 A comparison between our initial dataset and processed dataset.
The data points are narrowed down, eliminating all the noises that were
present at the beginning.

> Our module narrowed down 37 peaks from a dataset of 100000 data points
> as shown in Fig.4. The original dataset is represented using yellow
> color, and the cleaned one is represented using black. Additionally,
> we have tested our module with an additional four datasets. In this
> case, for each of the datasets, we have compared two consecutive
> stages for all the phases. The stages are Initial, Cluster, 1st
> Wavelet, 1st Median, 2nd Wavelet, 2nd Median, and Final. We
> represented the data by superimposing the plots on top of one another.
> Each of the six graphs represents a comparison between the two phases
> of data denoising. For any graph, the comparisons are between (a) raw
> data and data after clustering, (b) data after clustering and data
> after first wavelet analysis, (c) data after first wavelet analysis
> and data after first filtering, (d) data after first filtering vs.
> data after second wavelet analysis, (e) data after second wavelet
> analysis and data after second filtering and (f) data after second
> filtering and final value. Also, unlike in our initial test where we
> have plotted a data points vs. fuel graph, we have drawn a graph where
> X-axis corresponds to date and time, and Y-axis corresponds to fuel
> value. Their graphical representations are presented from Fig.5 to
> Fig.8.presented from Fig.5 to Fig.8.

![A picture containing text, map Description automatically
generated](media/image9.png){width="6.5in" height="5.767361111111111in"}

Fig.5 (a)-(c) One notable point here is most of the data is cleaned in
the first three phases. (d) There seems to be no effect of the second
wavelet after the first filtering is performed. (e) Second, filtering
has caused small changes in the dataset. (f) Changes are significant as
compared to immediately previous ones.

![A close up of a map Description automatically
generated](media/image10.png){width="6.5in"
height="5.767361111111111in"}

Fig.6 (a)-(c) Noise in prominent and heavy cleaning is performed.
(d)-(e) Dissimilar to the previous graph, the state of the graph remains
almost the same. Hence, there seems to be a small effect of second
wavelet and second filtering after the first filtering is performed. (f)
Final peak validation changes the graph by a small margin.

![A close up of a map Description automatically
generated](media/image11.png){width="6.5in"
height="5.728472222222222in"}

Fig.7 (a)-(c) Another notable point is that noise is more condensed
along with the consumption rate, which is shown in the graph by a
downward trend. (d) Similar to Fig.6, the difference is minimal after
the implementation of wavelet for the second time. (e)-(f) Cleaning
effect is visible in the downward trend.

![A close up of a map Description automatically
generated](media/image12.png){width="6.5in"
height="5.728472222222222in"}

Fig.8 (a)-(c) Fluctuations in fuel level is much prominent as compared
to the previous dataset. (d)-(e) Although, state of the graph remains
almost the same, (f) in the last stage, due to the peak validation
method, we see changes in the final dataset as compared to the previous
adjacent graph.

4.  **Conclusion**

> In conclusion, the real dataset (on which we have worked) has
> contained severe noise, which may have occurred due to poor street
> condition of some roads of Bangladesh, noisy dynamics, mobile
> environment and so on. The proposed module in this work, regardless of
> the intensity of data noise, is capable of detecting peaks without
> removing the noise in the first place. The reason behind the
> capability of doing that: different methods have been arranged in such
> a way that it can store the data before sending it to the next phase
> or process. For instance, data generated from clustering is stored
> first before sending it to the following phase to perform wavelet
> transform. This strategy allows the system to compare the values of
> the datasets retrieved from two adjacent processes. Before removing
> the noise, this entire task is conducted using multiple peak validity
> and peak detection methods. Consequently, the system works more
> efficiently as it is independent of the noise percentage. So
> regardless of the percentage of noise present in the datasets, peaks
> have been detected. However, after successful peak detection, the
> noise has been gotten removed and the correct value of the consumption
> rate has also been generated. So, this system can isolate noise and
> peak first; and then remove noises from the rest of the datasets,
> which provides not only correct peak values but also consumption
> values as well. Since this is a two layer-based noise removal system,
> at first, peaks have been separated from the noisy dataset. Then, the
> consumption rate is deduced using these separated peaks. This approach
> helps to maintain the integrity of the peaks. This type of design
> allows us to minimize data loss during noise removal, which has not
> been reported previously in the presence of bizarre data patterns i.e.
> constant values, accuracy subsides. Without such extreme cases,
> accuracy may increase a lot. Our work can be implemented in any of the
> industrial fields, where exist the complicated issues of fuel
> measurement i.e. vehicle industry, fuel station, power plant and so
> on. As long as any time-series data distribution resembles our dataset
> distribution, the module can work efficiently to reduce noise and
> produce correct values.
>
> **Acknowledgement**
>
> M.R.C. Mahdy acknowledges Dr. K.M. Masum Habib and Mr. Mahmudul Hasan
> Sohag of Pi-Labs Bangladesh Limited for several important discussions.
> M.R.C. Mahdy also acknowledges Mr. Saikat Chandra Das for several
> important discussions.

**Conflict of Interest**

Authors declare no competing financial interest.

**References**

1.  Kollem, S., Reddy, K. R. L., & Rao, D. S. (2019). A Review of Image
    Denoising and Segmentation Methods Based on Medical
    Images. *International Journal of Machine Learning and
    Computing*, *9*(3).

2.  Gondara, L. (2016, December). Medical image denoising using
    convolutional denoising autoencoders. In *2016 IEEE 16th
    International Conference on Data Mining Workshops (ICDMW)* (pp.
    241-246). IEEE.

3.  Singh, A., & Verma, R. S. (2017). An efficient non-local approach
    for noise reduction in natural images. *International Journal of
    Advanced Research in Computer Science*, *8*(5).

4.  Kaur, C., & Bansal, N. (2016). De-Noising Medical Images Using Low
    Rank Matrix Decomposition NN and SVM. *(IJCSIT) International
    Journal of Computer Science and Information Technologies*, *7*(1),
    45-48.

5.  Saihood, A. A. (2014). Image Denoising using Neural Network with SVM
    (Support Vector Machine) and LDA (Linear Discriminant
    Analysis). *Int. J. Comput. Sci. Inf. Technol.*, *5*(4), 5363-5367.

6.  Wu, J., Huang, D. Y., Xie, L., & Li, H. (2017, August). Denoising
    Recurrent Neural Network for Deep Bidirectional LSTM Based Voice
    Conversion. In *INTERSPEECH* (pp. 3379-3383).

7.  Yang, Q., Yan, P., Zhang, Y., Yu, H., Shi, Y., Mou, X., & Wang, G.
    (2018). Low-dose CT image denoising using a generative adversarial
    network with Wasserstein distance and perceptual loss. *IEEE
    transactions on medical imaging*, *37*(6), 1348-1357.

8.  Miranda, A. L., Garcia, L. P. F., Carvalho, A. C., & Lorena, A. C.
    (2009, June). Use of classification algorithms in noise detection
    and elimination. In *International Conference on Hybrid Artificial
    Intelligence Systems* (pp. 417-424). Springer, Berlin, Heidelberg.

9.  Germain, F. G., Chen, Q., & Koltun, V. (2018). Speech denoising with
    deep feature losses. *arXiv preprint arXiv:1806.10522*.

10. Singla, E. M., & Singh, M. H. (2015). Paper on frequency based audio
    noise reduction using butterworth, Chebyshev & Elliptical
    filters. *Int. J. Recent Innov. Trends Comput. Commun*, *3*,
    5989-5995.

11. Haque, M., & Bhattacharyya, K. (2018). Speech Background Noise
    Removal Using Different Linear Filtering Techniques. In *Advanced
    Computational and Communication Paradigms* (pp. 297-307). Springer,
    Singapore.

12. Welk M., Bergmeister A., Weickert J. (2005) Denoising of Audio Data
    by Nonlinear Diffusion. In: Kimmel R., Sochen N.A., Weickert J.
    (eds) Scale Space and PDE Methods in Computer Vision.
    Scale-Space 2005. Lecture Notes in Computer Science.

13. Magsi, H., Sodhro, A. H., Chachar, F. A., & Abro, S. A. K. (2018,
    March). Analysis of signal noise reduction by using filters.
    In *2018 International Conference on Computing, Mathematics and
    Engineering Technologies (iCoMET)* (pp. 1-6). IEEE.

14. Magondu, S. (2016). *Using Neural Networks to reduce noise in
    internet of things data streams* (Doctoral dissertation, University
    of Nairobi).

15. Agresti, G., Schaefer, H., Sartor, P., & Zanuttigh, P. (2019).
    Unsupervised domain adaptation for ToF data denoising with
    adversarial learning. In *Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition* (pp. 5584-5593).

16. He, Q., Wang, X., & Zhou, Q. (2014). Vibration sensor data denoising
    using a time-frequency manifold for machinery fault
    diagnosis. *Sensors*, *14*(1), 382-402.

17. Manmohan, Gupta, S., & Kuldeep. (2015). An Novel Approach for image
    denoising using Wavelet Transforms. International Journal of
    Engineering Research and Management (IJERM), 02(1), 142-146.

18. Mythili, C., & Kavitha, V. (2011). Efficient technique for color
    image noise reduction. *The research bulletin of Jordan,
    ACM*, *1*(11), 41-44.

19. Koskinen, K., Auvinen, P., Björkroth, K. J., & Hultman, J. (2015).
    Inconsistent denoising and clustering algorithms for amplicon
    sequence data. *Journal of Computational Biology*, *22*(8), 743-751.

20. Chatterjee, P., & Milanfar, P. (2009). Clustering-based denoising
    with locally learned dictionaries. *IEEE transactions on Image
    Processing*, *18*(7), 1438-1451.

21. Mahdy, M. R. C., Rivy, H. M., Jony, Z. R., Alam, N. B., Masud, N.,
    Al Quaderi, G. D., & Rahman, M. S. (2020). Dielectric or plasmonic
    Mie object at air--liquid interface: The transferred and the
    traveling momenta of photon. *Chinese Physics B*, *29*(1), 014211.

22. Chowdhury, R., Mahdy, M. R. C., Alam, T. N., Al Quaderi, G. D., &
    Rahman, M. A. (2020). Predicting the stock price of frontier markets
    using modified Black--Scholes Option pricing model and machine
    learning. *Physica A: Statistical Mechanics and its
    Applications*, 124444.

23. Chowdhury, R., Rahman, M. A., Rahman, M. S., & Mahdy, M. R. C.
    (2020). An approach to predict and forecast the price of
    constituents and index of cryptocurrency using machine
    learning. *Physica A: Statistical Mechanics and its
    Applications*, 124569.

24. Ahmed, Nahian, M. Nazmul Islam, Ahmad Saraf Tuba, M. R. C. Mahdy,
    and Mohammad Sujauddin. \"Solving visual pollution with deep
    learning: A new nexus in environmental management.\" *Journal of
    environmental management* 248 (2019): 109253.

25. Hambal, A. M., Pei, Z., & Ishabailu, F. L. (2017). Image noise
    reduction and filtering techniques. *International Journal of
    Science and Research (IJSR)*, *6*(3), 2033-2038.

26. Ding, S., Zhang, L., & Zhang, Y. (2010, April). Research on spectral
    clustering algorithms and prospects. In *2010 2nd International
    Conference on Computer Engineering and Technology* (Vol. 6, pp.
    V6-149). IEEE.

27. Murtagh, F. (2011). Hierarchical Clustering. Computing Research
    Repository 633-635.

28. Sudha, S., Suresh, G. R., & Sukanesh, R. (2009). Speckle noise
    reduction in ultrasound images by wavelet thresholding based on
    weighted variance. *International journal of computer theory and
    engineering*, *1*(1), 7.

29. Aggarwal, R., Singh, J. K., Gupta, V. K., Rathore, S., Tiwari, M., &
    Khare, A. (2011). Noise reduction of speech signal using wavelet
    transform with modified universal threshold. *International Journal
    of Computer Applications*, *20*(5), 14-19.

30. Üstündağ, M., Şengür, A., Gökbulut, M., & Ata, F. (2013).
    Performance comparison of wavelet thresholding techniques on weak
    ECG signal denoising. *Przegląd Elektrotechniczny*, *89*(5), 63-66.

31. Boateng, K. O., Asubam, B. W., & Laar, D. S. (2012). Improving the
    effectiveness of the median filter.

32. Gao, Jianbo & Sultan, Hussain & Hu, Jing & Tung, Wen-Wen. (2010).
    Denoising Nonlinear Time Series by Adaptive Filtering and Wavelet
    Shrinkage: A Comparison. Signal Processing Letters, IEEE. 17. 237 -
    240. 10.1109/LSP.2009.2037773.

33. Boto Giralda, Daniel & Díaz-Pernas, Francisco & González-Ortega,
    David & Díez, Jose & Antón-Rodríguez, Miriam & Martínez Zarzuela,
    Mario & De la Torre Díez, Isabel. (2010). Wavelet-Based Denoising
    for Traffic Volume Time Series Forecasting with Self-Organizing
    Neural Networks. Comp.-Aided Civil and Infrastruct. Engineering. 25.
    530-545. 10.1111/j.1467-8667.2010.00668.x.

34. Chou, Chien-Ming. (2011). A Threshold Based Wavelet Denoising Method
    for Hydrological Data Modelling. Water Resources Management. 25.
    1809-1830. 10.1007/s11269-011-9776-3.

35. Pandey, Brij & Tiwari, Harinarayan & Khare, Deepak. (2017). Trend
    analysis using discrete wavelet transform (DWT) for long-term
    precipitation (1851--2006) over India. Hydrological Sciences
    Journal/Journal des Sciences Hydrologiques. 62.
    10.1080/02626667.2017.1371849.

36. Musial, Jan & Verstraete, Michel & Gobron, Nadine. (2011). Comparing
    the effectiveness of recent algorithms to fill and smooth incomplete
    and noisy time series. Atmospheric Chemistry and Physics
    Discussions. 11. 10.5194/acpd-11-14259-2011.

37. Du, Pan & Kibbe, Warren & Lin, Simon. (2006). Improved peak
    detection in mass spectrum by incorporating continuous wavelet
    transform-based pattern matching. Bioinformatics (Oxford,
    England). 22. 2059-65. 10.1093/bioinformatics/btl355.

38. Palshikar, Girish. (2009). Simple Algorithms for Peak Detection in
    Time-Series.
