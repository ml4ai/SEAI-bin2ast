"""
Code for displaying and saving images
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def save_responsibility_map(inp, output, attention, destination_path):
    """
    save the responsibility map to given destination path
    :param inp: input sequence
    :param output: output sequence
    :param attention: attention matrix of shape [batch_size, n_heads, out_seq, in_seq]
    :param destination_path: destination path to save the image
    :return: None
    """

    plt.figure(figsize=(12, 12))
    ax = plt.axes()
    _attention = attention.squeeze(0).cpu().detach().numpy()
    _attention = np.mean(_attention, axis=0)
    ax.matshow(_attention, cmap='bone')
    ax.tick_params(labelsize=12, bottom=False)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in inp] + ['<eos>'], rotation=90)
    ax.set_yticklabels([''] + output)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(destination_path / 'responsibility.png')
    plt.close()
