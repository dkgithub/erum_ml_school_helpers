# coding: utf8

"""
Helpful plotting functions.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

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
        # l = l[:, 1]
        # p = p[:, 1]

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
    """ L = history()
        L.append(a,b,c)
        L = history('loss','val_loss','av_acc','acc')
        L.append(a,b,c,d)
        sorry, names hardwired
        L.plotLearningCurves()
    """
    def __init__(self,*names):
        self.history={}
        if len(names)==0:
            #default set
            names=('loss','val_loss','acc')
        for name in names:
            self.history[name]=[] #a list of values
    def append(self,*values):
        if len(values)!=len(self.history): 
            print('not enough values. Expect ',len(self.history))
            return 0
        for i,k in enumerate(self.history.keys()):
            self.history[k].append(values[i])
        return i+1
    def plotLearningCurves(self,start=0,stop=None):
        if stop==None:stop=len(self.history['loss'])
        plt.figure(figsize=(10,5))
        # losses
        plt.subplot(1,2,1)
        plt.plot(self.history['loss'][start:stop])
        plt.plot(self.history['val_loss'][start:stop])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper right')
        # accuracy plot
        ax = plt.subplot(1,2,2)
        ax.set_ylim([50, 95])
        plt.plot(self.history['acc'])
        if len(self.history)==4:
            plt.plot(self.history['av_acc'])
            plt.legend(['train','validation'], loc='lower right')
        else: plt.legend(['validation'], loc='lower right')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
        minValid = np.argmin(self.history['val_loss'])
        minTrain = np.argmin(self.history['loss'])
        maxAcc   = np.argmax(self.history['acc'])
        print(f"min train loss {self.history['loss'][minTrain]:5.3f} at ep. {minTrain}")
        print(f"min valid loss {self.history['val_loss'][minValid]:5.3f} at ep. {minValid}")
        print(f"best accurracy {self.history['acc'][maxAcc]:6.2f} at ep. {maxAcc}")

def plotLearningCurvesSkorch(history,start=0,stop=None):
  """
  Usage plotLearningCurvesSkorch(history) with skorch net.history object
        plotLearningCurvesSkorch(history,start)
        plotLearningCurvesSkorch(history,start,stop)

  """
  if stop==None:stop=len(history)
  plt.figure(figsize=(10,5))
  # losses
  plt.subplot(1,2,1)
  plt.plot(history[start:stop,'train_loss'])
  plt.plot(history[start:stop,'valid_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train','validation'], loc='upper right')
  # accuracy plot
  ax = plt.subplot(1,2,2)
  ax.set_ylim([50, 95])
  plt.plot(100*history[start:stop,'accuracy'])
  plt.plot(100*history[start:stop,'valid_acc'])
  plt.legend(['train','validation'], loc='lower right')
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.show()
  minValid = np.argmin(history[:,'valid_loss'])
  minTrain = np.argmin(history[:,'train_loss'])
  maxAcc   = np.argmax(history[:,'valid_acc'])
  print(f"min train loss {history[:,'train_loss'][minTrain]:5.3f} at ep. {minTrain}")
  print(f"min valid loss {history[:,'valid_loss'][minValid]:5.3f} at ep. {minValid}")
  print(f"best valid accurracy {history[:,'accuracy'][maxAcc]:6.2f} at ep. {maxAcc}")
