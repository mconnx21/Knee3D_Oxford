from matplotlib import pyplot as plt
import numpy as np


def entropy_by_angle(patient, side):
    entropy_stats= np.load("results\\"+patient+"_"+side+"_angleEntropies.npz")['e']
    entropies = []
    angles = range(336,380,2) #this is artificial and you need to fix how you're saving stats

    for entropy, scale, height in entropy_stats:
        entropies.append(entropy)
    print(angles)
    print(entropies)
    plt.plot(angles, entropies)
    plt.show()

entropy_by_angle("9964731", "LEFT")
