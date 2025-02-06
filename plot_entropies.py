from matplotlib import pyplot as plt
import numpy as np


def entropy_by_angle(patient, side):
    entropy_stats= np.load("results\\"+patient+"_"+side+"_angleEntropies.npz")['e']
    entropies = []
    angles =[340, 344, 348,352,356,360,364,368, 372, 376] #this is artificial and you need to fix how you're saving stats

    for entropy, scale, height in entropy_stats:
        entropies.append(entropy)
    
    plt.plot(angles, entropies)
    plt.show()

entropy_by_angle("9947240", "LEFT")
