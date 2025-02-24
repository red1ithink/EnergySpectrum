from Function import *

##### usage example #####
# from K_eps import keps

# times1, max_ks1 = keps(files, "[x1]")
# times2, max_ks2 = keps(files2, "[x10]")
# times3, max_ks3 = keps(files3, "[x0.5]")
# times4, max_ks4 = keps(files4, "[x2]")


def keps(files, name=""):

    times = []
    max_ks = []

    for file in files:
        k, e_k = get_ek(file)
        label = float(file.split('/')[-1].split('_')[0])
        max_index = np.argmax(e_k)
        max_k = k[max_index]

        times.append(label)
        max_ks.append(max_k)

        print(f"최대 e_k 값: {e_k[max_index]}, 해당 k 값: {max_k}, t: {label}s")

    times, max_ks = zip(*sorted(zip(times, max_ks)))

    plt.figure(figsize=(8, 6))
    plt.plot(times, max_ks, marker='o', linestyle='-')
    plt.xlabel('time (s)')
    plt.ylabel('k_eps')
    plt.title(f'k_eps {name}')
    plt.grid()
    plt.show()

    return list(times), list(max_ks)
