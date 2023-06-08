"""
Plot the learning curve for a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/8/2023 10:35 AM CT
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')

    args = vars(parser.parse_args())

    with open("%s/model_%d/model_%d_history.csv" % (args['model_dir'], args['model_number'], args['model_number']), 'rb') as f:
        history = pd.read_csv(f)

    model_properties = pd.read_pickle(f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl")

    # Model properties
    try:
        loss = model_properties['loss']
    except KeyError:
        loss = model_properties['loss_string']

    try:
        metric_string = model_properties['metric']
    except KeyError:
        metric_string = model_properties['metric_string']

    if model_properties['deep_supervision']:
        train_metric = history['sup1_Softmax_%s' % metric_string]
        val_metric = history['val_sup1_Softmax_%s' % metric_string]
    else:
        train_metric = history[metric_string]
        val_metric = history['val_%s' % metric_string]

    if 'fss' in loss.lower():
        loss_title = 'Fractions Skill Score (loss)'
    elif 'bss' in loss.lower():
        loss_title = 'Brier Skill Score (loss)'
    elif 'csi' in loss.lower():
        loss_title = 'Categorical Cross-Entropy'
    else:
        loss_title = None

    if 'fss' in metric_string:
        metric_title = 'Fractions Skill Score'
    elif 'bss' in metric_string:
        metric_title = 'Brier Skill Score'
    elif 'csi' in metric_string:
        metric_title = 'Critical Success Index'
    else:
        metric_title = None

    min_val_loss_epoch = np.where(history['val_loss'] == np.min(history['val_loss']))[0][0] + 1

    num_epochs = len(history['val_loss'])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    axarr = axs.flatten()

    plt.text(x=0, y=-0.02, s='Epoch %d' % min_val_loss_epoch, color='black', va='center', fontdict=dict(fontsize=11, fontweight='bold'))
    plt.text(x=0, y=-0.03, s='Training/Validation loss: %.4e, %.4e' % (history['loss'][min_val_loss_epoch - 1], history['val_loss'][min_val_loss_epoch - 1]), color='black', va='center', fontdict=dict(fontsize=11))
    plt.text(x=0, y=-0.04, s='Training/Validation metric: %.4f, %.4f' % (train_metric[min_val_loss_epoch - 1], val_metric[min_val_loss_epoch - 1]), color='black', va='center', fontdict=dict(fontsize=11))

    axarr[0].set_title(loss_title)
    axarr[0].plot(np.arange(1, num_epochs + 1), history['loss'], color='blue', label='Training loss')
    axarr[0].plot(np.arange(1, num_epochs + 1), history['val_loss'], color='red', label='Validation loss')
    axarr[0].set_xlim(xmin=0, xmax=num_epochs + 1)
    axarr[0].set_xlabel('Epochs')
    axarr[0].legend(loc='best')
    axarr[0].grid()
    axarr[0].set_yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions appear as very sharp curves.

    axarr[1].set_title(metric_title)
    axarr[1].plot(np.arange(1, num_epochs + 1), train_metric, color='blue', label='Training')
    axarr[1].plot(np.arange(1, num_epochs + 1), val_metric, color='red', label='Validation')
    axarr[1].set_xlim(xmin=0, xmax=num_epochs + 1)
    axarr[1].set_ylim(ymin=0)
    axarr[1].set_xlabel('Epochs')
    axarr[1].legend(loc='best')
    axarr[1].grid()

    plt.tight_layout()
    plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (args['model_dir'], args['model_number'], args['model_number']), bbox_inches='tight')
    plt.close()
