import numpy as np
import matplotlib.pyplot as plt

path = './res_new_damping00/'
n_iter = 10
x = [10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]

for p in [0.1, 0.2, 0.3, 0.4, 0.45, 0.46, 0.47, 0.48]:
    list_ave = []
    for traces in x:
        list_correct = []
        for i in range(n_iter):
            with open(path + f'{p}_{traces}/{i}.out', 'r') as f:
                lines = f.readlines()
                last = lines[-1]
                #print(p, traces, i, last)
                correct = int(last.split("correct=")[1].split("/")[0])
                list_correct.append(correct)
        ave = np.mean(list_correct)
        list_ave.append(ave)

    plt.plot(x, list_ave, label=f'p={p}')
    plt.xlabel("Traces")
    plt.ylabel("Corrrect")
    plt.title("Correct Coefficients")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./res.pdf')
