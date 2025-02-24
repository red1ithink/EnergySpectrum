# Two dimensional Turbulent Energy Spectrum

## Features

- **Data Analysis:**  
  Import and process energy spectrum data using Python libraries.
  
- **Change Point Detection:**  
  Utilize advanced algorithms (such as the PELT algorithm with a custom cost function) to segment the spectrum into distinct scaling regions.
  
- **Segment Fitting:**  
  Apply linear regression on each segment to estimate power-law exponents and perform statistical tests to compare against theoretical models.


## Getting Started

### Prerequisites

- [Python 3.12.8]


## Usage

### Running the Analysis

The main analysis script performs the following steps:

1. **Data Loading:**  
   Loads $$\( k \) and $$\( E(k) \) data from specified files using a custom `get_ek` function.

2. **Preprocessing:**  
   - Applies a logarithmic transformation to both $$\( k \) and $$\( E(k) \) (using natural logarithm via `np.log`).
   - Smooths the log-transformed data with a Savitzky–Golay filter to reduce noise.
   - Interpolates the smoothed data uniformly in log-space to ensure stable segmentation and regression.

3. **Change Point Detection:**  
   Utilizes the PELT algorithm with a custom cost function (based on least-squares linear regression error) to detect change points in the log–log data.

4. **Segment Fitting and Statistical Analysis:**  
   For each detected segment, the project performs linear regression to extract the slope (indicating the power-law exponent) and applies t-tests to evaluate whether the estimated slopes match theoretical expectations (e.g., $$\(-5/3\)$$, $$\(-4\)$$).

5. **Visualization:**  
   Generates log–log plots of the original spectrum along with the fitted segments, using distinct colors and line styles for clear differentiation. Segment indices and corresponding $$\( k \)$$-ranges are also printed to the console.

To run the analysis, simply execute the main script:

```bash
python main.py
```

Feel free to adjust parameters such as filter window length, interpolation resolution, and penalty values within the code to better suit your specific dataset.

## Methodology

### Data Transformation

- **Logarithmic Transformation:**  
  By converting $$\( k \)$$ and $$\( E(k) \)$$ to their natural logarithms, a power-law relationship  
  $$E(k) \sim k^n$$
    
  becomes linear:  
  $$ln E(k) = n \ln k + \ln C$$
  
- **Savitzky–Golay Smoothing:**  
  This filter applies a local polynomial fit over a moving window, reducing noise while preserving key spectral features.

- **Uniform Interpolation in Log-Space:**  
  Although the original $$\( k \)$$ data might be uniformly spaced, their logarithms typically are not. Uniformly resampling the log-transformed data ensures a consistent grid for reliable change point detection and regression analysis.

### Change Point Detection with PELT

- **Custom Cost Function:**  
  The project defines a cost function that, for any segment $$([s, e)\)$$, computes the sum of squared errors (SSE) from a linear fit:
  $${SSE}(s,e) = \sum_{i=s}^{e-1} \left[\ln E(k_i) - (m\,\ln k_i + b)\right]^2.$$
  
- **PELT Algorithm:**  
  The Pruned Exact Linear Time (PELT) algorithm minimizes the total cost (segment costs plus a penalty for additional change points) to determine the optimal segmentation of the spectrum.

### Segment-wise Analysis

- **Linear Regression:**  
  Each segment is analyzed via linear regression to estimate the slope $$\( m \)$$ (which corresponds to the power-law exponent) and intercept $$\( b \)$$

- **Statistical Testing:**  
  T-tests are conducted on the regression results to compare the estimated slope against target theoretical values. The t-statistic is computed as:
  $$t = \frac{m - \text{target}}{se_m}$$
  where $$\( se_m \)$$ is the standard error of $$\( m \)$$

## Acknowledgements

[ruptures]([https://centre-borelli.github.io/ruptures/](https://github.com/deepcharles/ruptures?tab=readme-ov-file))

---
## Reference

[docs](https://github.com/red1ithink/flow/blob/main/2d_decay/docs/reference)
