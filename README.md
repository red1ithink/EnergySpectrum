# EnergySpectrum

EnergySpectrum of turbulence Analysis

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

[ruptures](https://centre-borelli.github.io/ruptures/)

---
## Reference

0. OpenFoam implementation of two-dimensional isotropic homogenous turbulence decay problem by ""Omar Sallam""
1. --https://turbustat.readthedocs.io/en/latest/--
2. Omer San <Generalized deconvolution procedure for structural modeling of turbulence>
3. G. I. Taylor <Statistical Theory of Turbulence>
4. Leonardo Campanelli <Dimensional analysis of two-dimensional turbulence>
5. Rahul Agrawal <Turbulent cascade, bottleneck and thermalized spectrum in hyperviscous flows>
6. Sylvain Serra <Turbulent kinetic energy spectrum in very anisothermal flows>
7. M. E. arachet <Small-Scale Dynamics of High-Reynolds-Number Two-Dimensional Turbulence>
8. Stephen B. Pop <Turbulent Flows>
9. Jannike Solsvik <Turbulence modeling in the wide energy spectrum: Explicit formulas for Reynolds number dependent energy spectrum parameters>
10. Axel Brandenburg <The Turbulent Stress Spectrum in the Inertial and Subinertial Ranges>
11. UCLA <2D Homogeneous Turbulence>
12. S Kida <The energy spectrum in the universal range of two-dimensional turbulence>
13. PHILIPPA O’NEILL <VELOCITY, VELOCITY GRADIENT AND VORTICITY STATISTICS OF GRID TURBULENCE OBTAINED USING DIGITAL CROSS-CORRELATION PIV>
14. H.J.H. Clerc <Two-dimensional turbulence in square and circular domains with no-slip walls>
15. https://github.com/sayin/Pyhton_LES_solver_2D_decaying_trubulence
16. Zuoli Xiao <Physical mechanism of the inverse energy cascade of two-dimensional turbulence: A numerical investigation>
17. Michael K. Rivera <The Inverse Energy Cascade of Two-Dimensional Turbulence>
18. G. Boffetta <Inverse energy cascade in two-dimensional turbulenc>
19. Philipp W. Schroeder <Divergence-free H(div)-FEM for time-dependent incompressible flows with applications to high Reynolds number vortex dynamics>
20. B. Andreotti <On probability distribution functions in turbulence. Part 1. A regularisation method to improve the estimate of a PDF from an experimental histogram>
21. Gregory Falkovich <Vorticity statistics in the direct cascade of two-dimensional turbulence>
22. Mirko Stumpo <Relating Intermittency and Inverse Cascade to Stochastic Entropy in Solar Wind Turbulence>
23. Alberto Vela-Mart´in <Entropy, irreversibility and cascades in the inertial range of isotropic turbulence>
24. Patrick Tabeling <Two-dimensional turbulence: a physicist approach>
25. Claude Basdevant <On the validity of the "Weiss criterion" in two-dimensional turbulence>
26. from github for ruptures == C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020.
27. Ronald W. Schafer <What Is a Savitzky-Golay Filter?>
28. G. BOFFETTA <Energy and enstrophy fluxes in the double cascade of two-dimensional turbulence>


This README.md provides an overview of EnergySpectrum, details its features and methodology, and explains how to install and use the project. Adjust it further to fit your specific project details if needed.
