# coding: utf8

"""
Helpful plotting functions.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_hist(
    arr,
    names=None,
    xlabel=None,
    ylabel="Entries",
    filename=None,
    legend_loc="upper center",
    **kwargs,
):
    kwargs.setdefault("bins", 20)
    kwargs.setdefault("alpha", 0.7)

    # consider multiple arrays and names given as a tuple
    arrs = arr if isinstance(arr, tuple) else (arr,)
    names = names or (len(arrs) * [""])

    # start plot
    fig, ax = plt.subplots()
    for arr, name in zip(arrs, names):
        ax.hist(arr, label=name, **kwargs)

    # legend
    if any(names):
        legend = ax.legend(loc=legend_loc)
        legend.get_frame().set_linewidth(0.0)

    # styles and custom adjustments
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if filename:
        fig.savefig(filename)

    return fig


def plot_roc(labels, predictions, names=None, xlim=(0.01, 1.0), ylim=(1.0, 1.0e2)):
    # start plot
    fig, ax = plt.subplots()
    ax.set_xlabel("True positive rate")
    ax.set_ylabel("1 / False positive rate")
    ax.set_yscale("log")
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    plots = []

    # treat labels and predictions as tuples
    labels = labels if isinstance(labels, tuple) else (labels,)
    predictions = predictions if isinstance(predictions, tuple) else (predictions,)
    names = names or (len(labels) * [""])
    for l, p, n in zip(labels, predictions, names):
        # linearize
        l = l[:, 1]
        p = p[:, 1]

        # create the ROC curve and get the AUC
        fpr, tpr, _ = roc_curve(l, p)
        auc = roc_auc_score(l, p)

        # apply lower x limit to prevent zero division warnings below
        fpr = fpr[tpr > xlim[0]]
        tpr = tpr[tpr > xlim[0]]

        # plot
        plot_name = (n and (n + ", ")) + "AUC {:.3f}".format(auc)
        plots.extend(ax.plot(tpr, 1. / fpr, label=plot_name))

    # legend
    legend = ax.legend(plots, [p.get_label() for p in plots], loc="upper right")
    legend.get_frame().set_linewidth(0.0)

    return fig

class history:
    """
    DK history object\
    L=history('loss','val_loss','acc','val_acc','')
    L=history()
    """
    def __init__(self,*names):
        self.history={}
        if len(names)==0:
            #default set
            names=('loss','val_loss','acc','val_acc','')
        self.name=names[-1]
        for name in names[:-1]:
            self.history[name]=[]#a list of values
            #print(name)
    def append(self,*values):
        if len(values)!=len(self.history):
            print('not enough values')
        for i,name in enumerate(self.history):
            self.history[name].append(values[i])

#L=history('loss','val_loss','acc','val_acc','')
#L=history()
#L.history['loss'].append(1)
#L.append(1,2,3,4)
#print(L.history['loss'],L.history['acc'],L.name,L)
def plotLearningCurves(*histObjs):
    """This function processes all histories given in the tuple.
    Left losses, right accuracies
    """
    # too many plots
    if len(histObjs)>10:
        print('Too many objects!')
        return
    # missing names
    for histObj in histObjs:
        if not hasattr(histObj, 'name'): histObj.name='?'
    names=[]
    # loss plot
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    # loop through arguments
    for histObj in histObjs:
        plt.plot(histObj.history['loss'])
        names.append('train '+histObj.name)
        plt.plot(histObj.history['val_loss'])
        names.append('validation '+histObj.name)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')

