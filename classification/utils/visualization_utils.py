"""Plot the confusion matrix in tensorboard."""

from textwrap import wrap
import itertools
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

class ConfusionMatrixHook(tf.train.SessionRunHook):

    def begin(self):
        self.predictions = []
        self.correct_labels = []

    def before_run(self, run_context):
        fetches = run_context.original_args.fetches
        new_fetches = {
            "predictions":fetches["predictions"],
            "labels":fetches["labels"],
            "display_names":fetches["display_names"],
            "global_step":fetches["global_step"],
            "summary_dir":fetches["summary_dir"]
            }
        return tf.train.SessionRunArgs(fetches=new_fetches)

    def after_run(self, run_context, run_values):
        self.predictions.append(run_values.results["predictions"])
        self.correct_labels.append(run_values.results["labels"])
        self.display_names = run_values.results["display_names"]
        self.global_step = run_values.results["global_step"]
        self.summary_dir = run_values.results["summary_dir"]

    def end(self, session):
        correct_labels = np.concatenate(self.correct_labels)
        predictions = np.concatenate(self.predictions)
        summary_dir = self.summary_dir.decode("utf-8")
        display_names = [name.decode("utf-8") for name in self.display_names]
        plot_confusion_matrix(correct_labels,
                              predictions,
                              display_names,
                              summary_dir,
                              self.global_step,
                              normalize=True)

def plot_confusion_matrix(correct_labels, 
                          predict_labels, 
                          display_names, 
                          summary_dir,
                          global_step,
                          title='Confusion matrix', 
                          tag = 'Confusion matrix', 
                          normalize=False):
    ''' 
    Parameters:
        correct_labels: These are your true classification categories.
        predict_labels: These are you predicted classification categories
        display_names: This is a list of labels which will be used to display the axix labels
        title='Confusion matrix': Title for your matrix
        tag: Name for the output summay
 

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    tick_marks = np.arange(len(display_names))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(display_names, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(display_names, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', 
                horizontalalignment="center", fontsize=4, 
                verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    _bio = BytesIO()
    fig.savefig(_bio, format="png")

    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag=tag,
            image=tf.Summary.Image(
                encoded_image_string=_bio.getvalue()))
    ])
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_writer.add_summary(summary, global_step)
    summary_writer.close()

    tf.logging.info('Confusion matrix written to summary with tag %s.', tag)