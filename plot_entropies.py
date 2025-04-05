from matplotlib import pyplot as plt
import numpy as np


def entropy_by_angle(patient, side, measure):
    entropy_stats= np.load("results\\"+patient+"\\"+measure+"\\angleEntropies.npz")['e']
    entropies = []
    angles = range(336, 380,1) #this is artificial and you need to fix how you're saving stats

    for entropy, scale, height in entropy_stats:
        entropies.append(entropy)
    print(angles)
    print(entropies)
    plt.plot(angles, entropies)
    plt.show()




def entropy_by_angle_report(path, angles, point=None):
    entropy_stats= np.load(path)['e']
    entropies = []

    for entropy, scale, height in entropy_stats:
        entropies.append(entropy)
    print(angles)
    print(entropies)

    plt.plot(angles, entropies)
    if point != None:
        entropy_of_point = entropies[angles.index(point)]
        plt.plot(point, entropy_of_point, "or")
        plt.annotate("True registration angle", xy=(point,entropy_of_point), xytext = (point +5, entropy_of_point + 1000), arrowprops =dict(facecolor = "black", shrink =0.05))
    plt.xlabel("Angle (°)")
    plt.ylabel("GMIe")
    plt.title("GMIe by Angle")
    plt.show()



#entropy_by_angle("9031930", "LEFT", "gmi_e")
#entropy_by_angle_report("data_for_report_plots\\9002817_gmie_by_angle.npz" , range(-24, 20,4), -12)

