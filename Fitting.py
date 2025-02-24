from Function import *

def fitting(files, k_i, nu, csv_filename):
    results = []
    k_diss_vals = []

    for file in files:
        k, e_k = get_ek(file)
        e_k[0] = 10e-21

        label = float(file.split('/')[-1].split('_')[0])

        # k_diss
        k_diss = kdiss(label, nu)
        k_diss_vals.append(k_diss)

        # k_eps
        k_eps_index = np.argmax(e_k)
        k_eps = k[k_eps_index]

        # ----- k_eps ~ k_i ----- #
        mask1 = (k >= k_eps) & (k <= k_i)
        k_vals1, E_vals1 = k[mask1], e_k[mask1]
        log_k1, log_E1 = np.log10(k_vals1), np.log10(E_vals1)
        slope1, intercept1, r_value1, _, _ = stats.linregress(log_k1, log_E1)

        # ----- k_i ~ k_diss ----- #
        mask2 = (k >= k_i) & (k <= k_diss)
        k_vals2, E_vals2 = k[mask2], e_k[mask2]
        log_k2, log_E2 = np.log10(k_vals2), np.log10(E_vals2)
        slope2, intercept2, r_value2, _, _ = stats.linregress(log_k2, log_E2)

        kolmogorov_slope1, kolmogorov_slope2 = -5/3, -4

        plt.figure(figsize=(10, 6))
        plt.loglog(k, e_k, '-', alpha=0.6, label="Data")
        plt.axvline(k_eps, color='g', linestyle='--', label='k_eps')
        plt.axvline(k_i, color='b', linestyle='--', label='k_i')
        plt.axvline(k_diss, color='r', linestyle='--', label='k_diss')

        plt.loglog(k_vals1, 10**(kolmogorov_slope1 * log_k1 + intercept1), '--g', label="Kolmogorov -5/3")
        plt.loglog(k_vals1, 10**(slope1 * log_k1 + intercept1), '--k', label=f"Fit slope={slope1:.2f}")
        plt.loglog(k_vals2, 10**(kolmogorov_slope2 * log_k2 + intercept2), '--g', label="Kolmogorov -4")
        plt.loglog(k_vals2, 10**(slope2 * log_k2 + intercept2), '--r', label=f"Fit slope={slope2:.2f}")

        plt.title(f'{csv_filename.replace(".csv", "")}, t = {label}s')
        plt.xlabel("k")
        plt.ylabel("E(k)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.show()

        deviation_eng = (abs(slope1 - kolmogorov_slope1) / abs(kolmogorov_slope1)) * 100
        deviation_est = (abs(slope2 - kolmogorov_slope2) / abs(kolmogorov_slope2)) * 100

        print(f"Energy(ENG) slope: {slope1:.3f}, Deviation from -5/3: {deviation_eng:.2f}%, R²: {r_value1**2:.3f}")
        print(f"Enstrophy(EST) slope: {slope2:.3f}, Deviation from -4: {deviation_est:.2f}%, R²: {r_value2**2:.3f}")

        results.append([label, k_eps, slope1, slope2, r_value1**2, r_value2**2, deviation_eng, deviation_est])

    df = pd.DataFrame(results, columns=["Label", "k_eps", "Slope_ENG", "Slope_EST", "R2_ENG", "R2_EST", "Deviation_ENG(%)", "Deviation_EST(%)"])
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

def average_fitting(files, k_i, nu, title="Averaged E(k)"):
    all_ek = []
    k_diss_vals = []

    for file in files:
        k, e_k = get_ek(file)
        e_k[0] = 1e-20
        all_ek.append(e_k)

        label = float(file.split('/')[-1].split('_')[0])
        k_diss_vals.append(kdiss(label, nu))

    all_ek = np.array(all_ek)
    mean_ek = np.mean(all_ek, axis=0)
    k_diss = np.mean(k_diss_vals)

    # k_eps
    k_eps_index = np.argmax(mean_ek)
    k_eps_val = k[k_eps_index]

    # ----- k_eps ~ k_i ----- #
    mask1 = (k >= k_eps_val) & (k <= k_i)
    k_vals1, E_vals1 = k[mask1], mean_ek[mask1]
    log_k1, log_E1 = np.log10(k_vals1), np.log10(E_vals1)
    slope1, intercept1, r_value1, _, _ = stats.linregress(log_k1, log_E1)

    # ----- k_i ~ k_diss ----- #
    mask2 = (k >= k_i) & (k <= k_diss)
    k_vals2, E_vals2 = k[mask2], mean_ek[mask2]
    log_k2, log_E2 = np.log10(k_vals2), np.log10(E_vals2)
    slope2, intercept2, r_value2, _, _ = stats.linregress(log_k2, log_E2)

    plt.figure(figsize=(10,6))
    plt.loglog(k, mean_ek, 'b-', alpha=0.8, label="Mean Spectrum")
    plt.axvline(k_eps_val, color='g', linestyle='--', label='k_eps')
    plt.axvline(k_i, color='b', linestyle='--', label='k_i')
    plt.axvline(k_diss, color='r', linestyle='--', label='k_diss')

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()