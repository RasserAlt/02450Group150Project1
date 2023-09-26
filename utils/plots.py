import matplotlib.pyplot as plt

def hist_plot(Xy, attribute_names):
    fig, sub_figs_2D = plt.subplots(4, 2)
    sub_figs = sub_figs_2D.flatten()

    # Plotting histogram for each continues attribute
    for i in range(len(attribute_names)):
        sub_fig = sub_figs[i]
        sub_fig.hist(Xy[:,i], density=True, alpha=0.5)
        sub_fig.set_title(attribute_names[i])

    fig.suptitle('Probability density of continues attributes', fontsize=12)
    plt.show()
    return None

def box_plot(Xy,attribute_names):
    fig, sub_figs = plt.subplots(1,len(attribute_names))

    # Plotting Boxplot for each continues attribute
    for i in range(len(attribute_names)):
        sub_fig = sub_figs[i]
        sub_fig.boxplot(Xy[:,i])
        sub_fig.set_title(attribute_names[i])
    plt.suptitle('Abalone continues attributes - Boxplot', fontsize=12)
    plt.show()

    return None