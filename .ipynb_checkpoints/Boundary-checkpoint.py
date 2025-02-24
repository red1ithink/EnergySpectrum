from Function import *
from DefineFiles import *

##### usage example #####
# from Boundary import boundary

# files_list = [files, files2, files3, files4]
# nu_values = [0.00001, 0.0001, 0.000005, 0.00002]

# times_dict, max_ks_dict, k_diss_vals_dict = process_files(files_list, nu_values)

# print(times_dict)
# print(max_ks_dict)
# print(k_diss_vals_dict)


def boundary(files_list, nu_values):
    times_dict = {}
    max_ks_dict = {}
    k_diss_vals_dict = {}

    for i, (files, nu) in enumerate(zip(files_list, nu_values), start=1):
        times = []
        max_ks = []
        k_diss_vals = []

        for file in files:
            k, e_k = get_ek(file)
            label = float(file.split('/')[-1].split('_')[0])
            max_index = np.argmax(e_k)
            max_k = k[max_index]
            times.append(label)
            max_ks.append(max_k)
            k_diss_vals.append(kdiss(label, nu))

        times_dict[f"times{i}"] = times
        max_ks_dict[f"max_ks{i}"] = max_ks
        k_diss_vals_dict[f"k_diss_vals{i}"] = k_diss_vals

    return times_dict, max_ks_dict, k_diss_vals_dict
