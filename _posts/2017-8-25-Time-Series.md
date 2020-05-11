---
layout: post
title: Short survey for time series modeling
---

Given an observed time series $\mathbf{x} = x_1, x_2,...,x_n$ the goal is to study past observations in order to capture the structure of the series in a mathematical model. The model is then used to generate future values for the series in order to make predictions.

Each instance $x_t$ of the observed time series is called a realization and is modeled as a random variable. We can only observe the time series at a finite number of realizations, but often it's convenient to model it as an infinite series. Since we have a collection of random variables, this denotes a stochastic process with statistical properties.

$$\mathbf{x} = \boldsymbol{\beta}^T\mathbf{z} + \mathbf{s} + \mathbf{w}$$

$\mathbf{x}$ is the observed time series, $\boldsymbol{\beta}$ is a vector of parameters for the model, $\mathbf{z}$ is the trend or level of the series - a function of the previous realizations being studied, $\mathbf{s}$ is the seasonal component modeled by a periodic function, $\mathbf{w}$ is a random component modeled as white noise (zero mean, constant variance).

We can estimate the model parameters using a maximum likelihood estimation (MLE). Keep in mind that we want to minimize the number of parameters while maximizing the degrees of freedom and not over-fitting the data.

## Measuring Dependencies
Some patterns of a time series are serially dependent, that is, a point $x_t$ depends on a previous point $x_s$. The auto-correlation function (ACF) $\rho (s,t) \in [-1,1]$ measures how well the value $x_t$ can be predicted from $x_s$ $\forall$  $t,s$. When we can predict $x_s$ perfectly from $x_t$, $|\rho (s,t)| = 1$. To calculate the ACF, we define the auto-covariance function $\gamma (s,t)$ which measures the linear dependence between observations $x_t$ and $x_s$. 

$$\gamma (s,t) = cov(x_s,x_t) = E[(x_s - \mu_s)(x_t - \mu_t)]$$

$$\rho (s,t) = \frac{\gamma (s,t)}{\sqrt{\gamma (s,s)\gamma (t,t)}}$$

It would be difficult to measure the dependence between the values of the series if the dependence structure is not regular or is changing at every time point. Hence, to achieve any meaningful statistical analysis of time series data, it's crucial that the mean and the auto-covariance functions satisfy the conditions of stationarity (more on that later). The time series should have constant mean and variance and the ACF should only depend on the time separation, or lag $h$, between observations $x_t$ and $x_s$ and not on the absolute location of the points along the series. Since, $\mu_s = \mu_t = \mu$ and $x_s = x_{t+h}$. The equations then become:

$$\gamma (h) = cov(x_t,x_{t+h}) = E[(x_t - \mu)(x_{t+h} - \mu)]$$

$$\rho (h) = \frac{\gamma (t+h,t)}{\sqrt{\gamma (t+h,t+h)\gamma (t,t)}} = \frac{\gamma (h)}{\gamma (0)}$$


Computing and visualizing the ACF will help us below when creating a model for the data.

## Differencing the data
How do we know whether the current data is stationary? The auto-correlation function can help indicate its behavior. For a stationary time series, the ACF will drop to zero relatively quickly, while the ACF of non-stationary data decreases slowly.

To convert the time series into a stationary process, we can perform differencing, denoted by $\Delta ^d$ where $d$ is the order of differencing. The goal is to eliminate the trend in the data. If the trend is linear, we take the first difference ($d$ = 1), or more generally, if the trend is a polynomial of degree $r$, we let $d = r$. The differenced series is a new series $\mathbf{z}$ such that $z_t = x_{t+d} - x_{t}$. Usually, if the level of the data shifts up and down, we use a differencing order of 1, when the slope shifts rapidly, we use a differencing order of 2. It's necessary to be careful not to over-difference because this may introduce dependence when none exists. 

Note that differencing is used to deal with a non-stationary mean, but if the variance is not stable, we can use a log transformation or a Box-cox transformation to stabilize it. These will normally distribute the data to reduce skewness and eliminate a changing variance.

## Modeling
### Auto-regressive (AR)
Auto-regressive models are based on the idea that the current value of the series, $x_t$, can be explained as a function of $p$ past values, $x_{t-1}, x_{t-2},...,x_{t-p}$ where $p$ is the lag which determines the number of steps into the past needed to forecast the current value. An auto-regressive model of order $p$, $AR(p)$, is of the form

$$x_t = \alpha + w_t + \sum_{i=1}^p \phi_i x_{t-i}$$

where $\alpha$ is a constant/intercept, $w_t$ is white noise, and $\phi_i$ determines how we weigh time step $x_{t-i}$.

### Moving average (MA) 
Independent of the auto-regressive model, each element in the series can also be affected by the past error that cannot be accounted for by $AR$. The moving average model aims to model the hidden noise and smooth the series. $MA(q)$ is of the form

$$x_t = \mu + w_t + \sum_{i=1}^q \theta_i w_{t-i}$$

where $\mu$ is the expected value of $x_t$ (often assumed to be zero).

### Auto-regressive integrated moving average (ARIMA)
Both the MA and AR models assume the model is stationary.
ARIMA is a combination of the AR and MA often written as $ARIMA(p,d,q)$ and accounts for non-stationary processes
where $p$ and $q$ are the number of AR and MA parameters, respectively, and $d$ is the order of differencing.

$$x_t = \alpha + w_t + \sum_{i=1}^p \phi_i x_{t-i} + \sum_{i=1}^q \theta_i w_{t-i}$$

To determine the values of $p$ and $q$, we look at the ACF and PACF plots of the differenced series. Generally, we can follow these rules:

|         | $AR(p)$                   | $MA(q)$                   | $ARIMA(p,q)$  |
|---------|:-------------------------:|:-------------------------:|:-------------:|
| ACF     | tails off                 | cuts off after lag $q$    |tails off      |
| PACF    | cuts off after lag $p$    | tails off                 |tails off      |

Note that ARIMA does not factor in the seasonal $\mathbf{s}$ component that the data may contain. However, there are extensions such as SARIMA that factors that in the seasonal to the model.

### Aside - Spectral Analysis
Any time series can be expressed as a combination of sine and cosine waves with differing periods and amplitudes. This is useful in identifying cycles and seasonal fluctuations of different lengths in the data. We do so by writing the time series in terms of its Fourier representation which is a linear transformation from the time to the frequency domain. When performing a FT, we assume that the frequencies are constant over time i.e. that it's a stationary series.

$$x_t = \sum_{k=1}^q a_k sin(2\pi\omega t) + b_k cos(2\pi\omega t)$$

where $a_k$ and $b_k$ are regression coefficients that tell us the degree to which the respective functions are correlated with the data, and $\omega$ is the frequency of oscillation. 

We want to find the frequencies with the greatest spectral densities, that is, the frequency regions, consisting of many adjacent frequencies, that contribute most to the overall periodic behavior of the series. This can be accomplished through smoothing using a moving average window. We must keep in mind the Heisenberg uncertainty principle. For signals with large periods, we want a bigger window of the signal in the frequency domain to estimate the frequency at the cost of losing temporal information. At high frequencies, we can use a small window to estimate the frequency so we gain temporal information. If frequencies change over time, we would use a wavelet transform which provides dynamic window sizes.

To summarize, spectral analysis will identify the correlation of sine and cosine functions of different frequencies with the observed data. If a large correlation (sine or cosine coefficient) is identified, we can conclude that there is a strong periodicity of the respective frequency in the data.

[Related post](https://github.com/vamaral1/fft-dwt-nasa-turbofan/blob/master/fft-dwt-nasa-turbofan.ipynb)

### Extension to multivariate time series
So far we've only discussed the univariate case. When we're trying to model multiple series and their interactions, we can extend some of these ideas to multivariate time series. In this case we have a $m \times n$ matrix $\mathbf{X}$ with each column containing a $n$ x $1$ vector representing a time series. In other words $\mathbf{X} = \mathbf{x_1}, \mathbf{x_2}, ..., \mathbf{x_m}$. 

## Summary

* Plot the data - Construct a time plot of the data and inspect for anomalies.
* Identify dependence - Calculate values of $d$, $p$, and $q$ based on the ACF and PACF. 
* Estimate the parameters - Use maximum likelihood estimation to fit the $MA(q)$ and $AR(p)$ models,  maximizing the likelihood of the observed series, given the parameter values. 
* Evaluation and diagnostics - A good model should be able to predict observations. To verify this we can calculate the model on part of the data and see if it predicts the rest of the data well. Lastly, the residuals should not contain noise nor should it have any trends. A way to measure goodness of fit is using the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC) - low values indicate a better fit. 

Some relevant R code is available [here](https://github.com/vamaral1/ts-anomaly-r).

## References
* Robert H Shumway and David S Stoffer. Time series analysis and its applications.
Springer Science & Business Media, 2013.
* Eric Zivot. Multivariate Time Series. <http://faculty.washington.edu/ezivot/econ584/notes/multivariatetimeseries.pdf>, 2006.
* Robert Nau. Statistical forecasting: notes on regression and time series analysis. <http://people.duke.edu/~rnau/411home.htm>, 2015.
