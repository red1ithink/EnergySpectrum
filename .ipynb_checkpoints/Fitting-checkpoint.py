from Function import *
from DefineFiles import *
from boundary import *


##### usage example ######
# from fitting import fitting

# files_list = [files, files2, files3, files4]
# k_diss_vals_list = [k_diss_vals1, k_diss_vals2, k_diss_vals3, k_diss_vals4]
# csv_filenames = ["[x1]results.csv", "[x10]results.csv", "[x0.5]results.csv", "[x2]results.csv"]

# for files, k_diss_vals, csv_filename in zip(files_list, k_diss_vals_list, csv_filenames):
#     fitting(files, k_diss_vals, csv_filename)


def fitting(files, k_diss_vals, csv_filename, k_i=18):
    results = []

    for file, k_diss in zip(files, k_diss_vals):
        k, e_k = get_ek(file)
        e_k[0] = 10e-21

        # k_eps
        k_eps_index = np.argmax(e_k)
        k_eps = k[k_eps_index]
        if k_eps > 12:
            continue

        # ----- k_eps ~ k_i ----- #
        k_min1, k_max1 = k_eps, k_i
        mask1 = (k >= k_min1) & (k <= k_max1)
        k_vals1, E_vals1 = k[mask1], e_k[mask1]
        
        log_k1, log_E1 = np.log10(k_vals1), np.log10(E_vals1)
        slope1, intercept1, r_value1, _, _ = stats.linregress(log_k1, log_E1)

        # ----- k_i ~ k_diss ----- #
        mask2 = (k >= k_i) & (k <= k_diss)
        k_vals2, E_vals2 = k[mask2], e_k[mask2]

        log_k2, log_E2 = np.log10(k_vals2), np.log10(E_vals2)
        slope2, intercept2, r_value2, _, _ = stats.linregress(log_k2, log_E2)

        kolmogorov_slope1, kolmogorov_slope2 = -5/3, -4

        label = file.split('/')[-1].split('_')[0]

        plt.figure(figsize=(10, 6))
        plt.loglog(k, e_k, '-', alpha=0.6, label="Data")
        plt.axvline(k_eps, color='g', linestyle='--')
        plt.axvline(k_i, color='b', linestyle='--')
        plt.axvline(k_diss, color='r', linestyle='--')

        plt.loglog(k_vals1, 10**(kolmogorov_slope1*log_k1 + intercept1), '--g', label="Kolmogorov -5/3")
        plt.loglog(k_vals1, 10**(slope1*log_k1 + intercept1), '--k', label=f"Fit slope={slope1:.2f}")
        plt.loglog(k_vals2, 10**(kolmogorov_slope2*log_k2 + intercept2), '--g', label="Kolmogorov -4")
        plt.loglog(k_vals2, 10**(slope2*log_k2 + intercept2), '--r', label=f"Fit slope={slope2:.2f}")

        plt.title(f'{csv_filename.replace(".csv", "")}, t = {label}s')
        plt.xlabel("k")
        plt.ylabel("E(k)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.show()

        # Deviation
        deviation_eng = (abs(slope1 - kolmogorov_slope1) / abs(kolmogorov_slope1)) * 100
        deviation_est = (abs(slope2 - kolmogorov_slope2) / abs(kolmogorov_slope2)) * 100

        print("------------------------")
        print(f"Energy(ENG) inertial range Slope: {slope1:.3f}")
        print(f"Deviation from -5/3: {deviation_eng:.2f}%")
        print(f"R² for Inverse Cascade: {r_value1**2:.3f}")
        print("------------------------")
        print(f"Enstophy(EST) inertial range Slope: {slope2:.3f}")
        print(f"Deviation from -4: {deviation_est:.2f}%")
        print(f"R² for EST: {r_value2**2:.3f}")

        results.append([
            label, k_eps, slope1, slope2, r_value1**2, r_value2**2, deviation_eng, deviation_est
        ])

    df = pd.DataFrame(results, columns=[
        "Label", "k_eps", "Slope_ENG", "Slope_EST", "R2_ENG", "R2_EST", "Deviation_ENG(%)", "Deviation_EST(%)"
    ])
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

