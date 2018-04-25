import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


#def show_frame(frame, bbox, fig_n):
def show_frame(frame, bbox,bbox1,bbox2, fig_n):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    rupper = patches.Rectangle((bbox1[0],bbox1[1]), bbox1[2], bbox1[3], linewidth=2, edgecolor='b', fill=False)
    rlower = patches.Rectangle((bbox2[0],bbox2[1]), bbox2[2], bbox2[3], linewidth=2, edgecolor='y', fill=False)
    
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    ax.add_patch(rupper)
    ax.add_patch(rlower)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf()


def show_crops(crops, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.uint8(crops[1,:,:,:]))
    ax3.imshow(np.uint8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_scores(scores, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)

