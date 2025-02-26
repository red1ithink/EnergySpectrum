import os
import re
import csv
import pyfftw
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from math import sqrt
from scipy.ndimage import sobel
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from scipy.fftpack import fft2, ifft2, fftshift
from visualize import create_animation 
from DefineFiles import *

#grid
N = 1024

def sep(data):
    df = pd.read_csv(data)
    ux = df['ux'].values
    uy = df['uy'].values
    
    return ux, uy
    
def resizing(ux, uy):
    global N
    ux_2d = ux.reshape((N, N))
    uy_2d = uy.reshape((N, N))
    return ux_2d, uy_2d

def get_ek(file):
    global N
    ux, uy = sep(file)
    ux,uy = resizing(ux, uy)
    k, e_k = ek(ux, uy, N)
    return k, e_k

def get_vorticity(file):
    
    data = pd.read_csv(file, header=0).values.flatten()
    data = data[~np.isnan(data)]
    data = data[0:N * N]
    label = file.split('/')[-1].split('_')[0]
    
    return data, label

def velocity_plotting(file):
    global N
    ux, uy = sep(file)
    label = file.split('/')[-1].split('_')[0]
    ux_2d = ux.reshape((N, N))
    uy_2d = uy.reshape((N, N))
    k_array, Ek_array = ek(ux_2d, uy_2d, N)
    Ek_array[0] = 10e-21
    label = file.split('/')[-1].split('_')[0]
    plt.loglog(k_array, Ek_array, label=f"{label}s")    

def compared_line():
    global N
    k_a = np.logspace(0, 1, 500)
    E_k_a = ((k_a)**-(5/3)) * 10e-1
    plt.loglog(k_a, E_k_a, linestyle='--', color='red', label="E(k) = k^-5/3")
    k_a_shift = k_a * 10
    E_k_a_2 = ((k_a)**-4) * 10e-3
    plt.loglog(k_a_shift, E_k_a_2, linestyle='--', color='black', label="E(k) = k^-4")

def compared_line2():
    global N
    k_a = np.logspace(0, 1, 5000)
    E_k_a = ((k_a)**-(5/3)) * 10e-1
    plt.plot(k_a, E_k_a, linestyle='--', color='red', label="E(k) = k^-5/3")
    k_a_shift = k_a * 10
    E_k_a_2 = ((k_a)**-4) * 10e-3
    plt.plot(k_a_shift, E_k_a_2, linestyle='--', color='black', label="E(k) = k^-4")

def ek(u, v, n_bins):

    Ny, Nx = u.shape
    Lx = 6.283
    Ly = 6.283
    
    dx = Lx / Nx
    dy = Ly / Ny

    u_prime = u - np.mean(u)
    v_prime = v - np.mean(v)

    U_hat = np.fft.fft2(u_prime)
    V_hat = np.fft.fft2(v_prime)

    PS = (np.abs(U_hat)**2 + np.abs(V_hat)**2) / (Nx * Ny)**2
    
    kx = 2.0*np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0*np.pi * np.fft.fftfreq(Ny, d=dy)
    
    KX, KY = np.meshgrid(kx, ky)
    
    K_mag = np.sqrt(KX**2 + KY**2)
    
    k_max = K_mag.max()
    
    bins = np.linspace(0, k_max, n_bins+1)
    
    E_k = np.zeros(n_bins)
    k_vals = np.zeros(n_bins)
    
    for i in range(n_bins):
        k_min = bins[i]
        k_max_ = bins[i+1]
        
        mask = (K_mag >= k_min) & (K_mag < k_max_)
        
        shell_sum = np.sum(PS[mask])
        
        k_mid = 0.5*(k_min + k_max_)
        
        E_k[i] = shell_sum
        k_vals[i] = k_mid

    return k_vals, E_k

def energy_spectrum_from_velocity(nx, ny, dx, u, v):
    """
    Computes the 2D isotropic energy spectrum from a 2D velocity field (u,v).

    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y directions (assumed equally spaced).
    dx     : float
        Grid spacing (assumed equal in x and y for simplicity).
    u, v   : 2D arrays of shape (nx, ny)
        Velocity components in physical space (with periodic boundaries).

    Returns
    -------
    en : 1D array of length n+1
        The isotropically averaged kinetic energy spectrum E(k) for k=0..n.
    n : int
        The maximum wavenumber index used in the radial binning.

    Notes
    -----
    1) This code follows the same radial-shell averaging strategy used in your
       original vorticity-based code, but adapted for velocity.
    2) The factor of `pi * k` in `es` is consistent with the usual 2D isotropic
       energy decomposition, i.e., E(k) = 2π k * (1/2)|û|^2 (on average).
       The division by `(nx*ny)` accounts for the unnormalized FFT in pyFFTW.
    3) We divide the final sum in each shell by `ic` (the number of modes in
       that shell) to produce an average spectral amplitude. That is the same
       convention used in the original code. 
    """

    # A small number to avoid zero in the wavenumbers
    epsilon = 1.0e-6

    # 1) Construct wavenumber arrays kx, ky for 2D periodic domain.
    kx = np.empty(nx)
    ky = np.empty(ny)

    # Positive frequencies: 0..(nx/2 - 1), negative frequencies: -(nx/2)..-1
    # 2π/(nx*dx) is the fundamental wavenumber spacing in x.
    kx[0:int(nx/2)] = 2.0*np.pi/(nx*dx) * np.arange(0, int(nx/2))
    kx[int(nx/2):nx] = 2.0*np.pi/(nx*dx) * np.arange(-int(nx/2), 0)

    # Do same for ky
    ky[0:int(ny/2)] = 2.0*np.pi/(ny*dx) * np.arange(0, int(ny/2))
    ky[int(ny/2):ny] = 2.0*np.pi/(ny*dx) * np.arange(-int(ny/2), 0)

    # Avoid dividing by zero at k=0
    kx[0] = epsilon
    ky[0] = epsilon

    # Make 2D mesh of wavenumbers
    kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')
    kk = np.sqrt(kx2d**2 + ky2d**2)  # radial wavenumber

    # 2) Allocate pyFFTW arrays for forward transform of u and v
    a_u = pyfftw.empty_aligned((nx, ny), dtype='complex128')
    b_u = pyfftw.empty_aligned((nx, ny), dtype='complex128')

    a_v = pyfftw.empty_aligned((nx, ny), dtype='complex128')
    b_v = pyfftw.empty_aligned((nx, ny), dtype='complex128')

    fft_u = pyfftw.FFTW(a_u, b_u, axes=(0,1), direction='FFTW_FORWARD')
    fft_v = pyfftw.FFTW(a_v, b_v, axes=(0,1), direction='FFTW_FORWARD')

    # Copy real-space velocity fields into a_u, a_v
    a_u[:] = u
    a_v[:] = v

    # Compute FFTs
    uf = fft_u()  # = \hat{u}(kx, ky)
    vf = fft_v()  # = \hat{v}(kx, ky)

    # 3) Define the spectral energy density ES(kx, ky).
    #    For 2D isotropic E(k), one often uses ~ pi*k*(|u_hat|^2 + |v_hat|^2)/ (nx*ny)^2
    #    The code below uses a style consistent with the original approach:
    #       factor pi,
    #       multiply by k (because E(k) ~ k * (|û|^2 + |v̂|^2)),
    #       divide by (nx*ny)^2 due to the unnormalized FFT in pyFFTW,
    #       then do radial bin-averaging.
    es = np.pi * ( (np.abs(uf)/(nx*ny))**2 + (np.abs(vf)/(nx*ny))**2 ) * kk

    # 4) Maximum integer radius in k-space for binning
    n = int(np.sqrt(nx*nx + ny*ny)/2.0) - 1

    # 1D array for final energy spectrum
    en = np.zeros(n+1)

    # 5) Radial bin average: for each integer k=1..n, average es in the shell k-0.5 < |k| < k+0.5
    for k in range(1, n+1):
        # Indices in the shell
        ii, jj = np.where((kk > (k-0.5)) & (kk <= (k+0.5)))
        ic = ii.size  # how many points in the shell

        if ic > 0:
            en[k] = np.sum(es[ii, jj]) / ic
        else:
            en[k] = 0.0

    return en, n

def energy_spectrum_from_vorticity(L, nx,ny,w):
    
    '''
    Computation of energy spectrum and maximum wavenumber from vorticity field
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    w : vorticity field in physical spce (including periodic boundaries)
    
    Output
    ------
    en : energy spectrum computed from vorticity field
    n : maximum wavenumber
    '''
    
    epsilon = 1.0e-6
    dx = L/nx

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[0:nx,0:ny]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
                    
        en[k] = en[k]/ic
        
    return en, n
##### energy dissipation code by KyunYoon Han #####
def ddy(f, dx = 6.283/N): #y방향 미분
  dudy = np.gradient(f,axis=0)
  dudy = dudy/dx
  return dudy

def ddx(f, dx = 6.283/N): #x방향 미분
  dudx = np.gradient(f,axis=1)
  dudx = dudx/dx
  return dudx

def energy_dissipation(u,v, nu): #에너지 소산율 계산
    dudx = ddx(u)
    dvdy = ddy(v)
    dudy = ddy(u)
    dvdx = ddx(v)
    nu = nu

    energy_dissipation = nu*(2*(dudx**2 + dvdy**2) + (dudy + dvdx)**2)
    return 2 * np.mean(energy_dissipation) 

def dissipation_rate_2d(U, V, nu):
    """
    input
    (U(x,y), V(x,y))
    nu
   
    output
    epsilon = 2 * nu * < S_ij S_ij >
    """
    L = np.pi * 2
    N = U.shape[0]
    dx = L / N

    U_prime = U - np.mean(U)
    V_prime = V - np.mean(V)

    # 중앙차분(주기경계조건)
    dUdx = (np.roll(U_prime, -1, axis=1) - np.roll(U_prime, 1, axis=1)) / (2 * dx)
    dUdy = (np.roll(U_prime, -1, axis=0) - np.roll(U_prime, 1, axis=0)) / (2 * dx)
    dVdx = (np.roll(V_prime, -1, axis=1) - np.roll(V_prime, 1, axis=1)) / (2 * dx)
    dVdy = (np.roll(V_prime, -1, axis=0) - np.roll(V_prime, 1, axis=0)) / (2 * dx)

    # 변형률 텐서 S_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)
    # (u1,u2)=(U,V), (x1,x2)=(x,y)
    # S_11 = ∂U/∂x,   S_22 = ∂V/∂y
    # S_12 = S_21 = 1/2 ( ∂U/∂y + ∂V/∂x )
    S11 = dUdx
    S22 = dVdy
    S12 = (dUdy + dVdx) *0.5

    # S_ij S_ij = S11^2 + 2*S12^2 + S22^2
    SijSij = S11**2 + 2.0*(S12**2) + S22**2 #* 1/4

    # dissipation rate: epsilon = 2 * nu * < S_ij S_ij >
    epsilon = 4 * nu * np.mean(SijSij)
    
    return epsilon

def enstrophy_diss(omega, nu):
    global N
    dx = 6.283 / N
    omega = omega.reshape((N, N))
    # ∇ω
    domega_dx = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dx)
    domega_dy = (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dx)
    
    # |∇ω|²
    grad_omega_squared = domega_dx**2 + domega_dy**2

    eta = nu * np.mean(grad_omega_squared)
    
    return eta
    
##k_v from <Statistical Physics Interpretation of Southern Ocean Mesoscale Turbulence>##
def k_v(omega, nu):
    eta = enstrophy_diss(omega, nu)
    kv = (eta/(nu)**3)**(1/6)
    return kv

##k_v 3d from <David Statistical Physics Interpretation of Southern Ocean Mesoscale Turbulence>
def kv_3d(eps, nu):
    kv = ((nu**3)/eps)**(-1/4)

    return kv

def kd_range(eps, nu):
    kd = (eps**(1/4))/(nu**(3/4))
    return kd

def enstrophy(file, nu, visualize=False):
    
    L = 6.283
    global N
    dx = L / N
    dy = dx
    
    data, label = get_vorticity(file)
    vorticity = data.reshape((N, N))
    
    # ∫ |ω|^2 dA
    enstrophy_value = np.sum(vorticity**2) * (dx * dy)

    if visualize:
        print(f"Visualizing enstrophy for {nu}")
        data_list = [vorticity]
        label = file.split('/')[-1].split('_')[0]
        create_animation(data_list, label, nu, num_files=1, save_filename=f"enstrophy {nu}.gif", interval_ms=500)
    
    return enstrophy_value, label

def kdiss(time, nu):
    L_diss = np.sqrt(nu*time)
    k_diss = 1/L_diss
    
    return k_diss

def energy_flux(u, v, L=6.283):
    N = u.shape[0]
    dx = dy = L / N
    
    u_hat = fft2(u)
    v_hat = fft2(v)
    
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2)
    
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    # Nonlinear terms
    Nx = u * du_dx + v * du_dy
    Ny = u * dv_dx + v * dv_dy
    
    # FFT of nonlinear terms
    Nx_hat = fft2(Nx)
    Ny_hat = fft2(Ny)
    
    # Energy transfer function
    T_k = -np.real(u_hat.conj() * Nx_hat + v_hat.conj() * Ny_hat)
    
    # Shell averaging
    k_bins = np.arange(1, 2048)
    T_k_binned = np.zeros_like(k_bins, dtype=np.float64)
    
    # Improved shell averaging with proper normalization
    for i, k in enumerate(k_bins):
        mask = (k_mag >= k) & (k_mag < k+1)
        if np.any(mask):
            T_k_binned[i] = np.sum(T_k[mask])  # Normalized by number of modes
    
    # Energy flux
    Pi_k = np.cumsum(T_k_binned[::-1])[::-1]
    
    return k_bins, Pi_k

def average_energy_flux(files, title, xlabel="k", ylabel=r"$\langle \Pi(k) \rangle$"):
    """
    여러 파일의 에너지 플럭스를 평균 내어 그래프를 출력하는 함수.

    Parameters:
        files (list): 분석할 파일 리스트
        title (str): 그래프 제목
        xlabel (str): X축 레이블
        ylabel (str): Y축 레이블

    Returns:
        list: 평균 에너지 플럭스의 부호가 변하는 k 값 리스트 (zero-crossings)
    """
    all_Pi_k = []
    all_k_bins = None

    plt.figure(figsize=(8, 6))

    for file in files:
        label = file.split('/')[-1].split('_')[0]
        ux, uy = sep(file)  
        u, v = resizing(ux, uy) 
        k_bins, Pi_k = energy_flux(u, v)

        if all_k_bins is None:
            all_k_bins = k_bins

        all_Pi_k.append(Pi_k)

    mean_Pi_k = np.mean(np.array(all_Pi_k), axis=0)
    plt.plot(all_k_bins, mean_Pi_k)
    plt.xscale('log')

    zero_crossing_indices = np.where(np.diff(np.sign(mean_Pi_k)))[0]
    zero_crossings_avg = [all_k_bins[idx] for idx in zero_crossing_indices]

    plt.axhline(y=0, color='black', linestyle='--', label='y = 0')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()

    print(f"Zero-crossing points for averaged energy flux ({title}):", zero_crossings_avg)
    return zero_crossings_avg
