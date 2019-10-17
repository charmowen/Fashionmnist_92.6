import matplotlib.pyplot as plt
import numpy as np
import torch

def vis_datasets(data_loader,text_labels):
    imgs, labels = next(iter(data_loader))
    imgs = imgs.numpy()
    labels = labels.numpy()
    fig = plt.figure(figsize = (25,8))
    for i in range(len(imgs)):
        ax = fig.add_subplot(3, 24/3,i+1,xticks=[],yticks = [])
        #plt.subplot(len(imgs) / 5 + 1, 5, i + 1)
        # print(imgs[i][0])
        if imgs[i].shape[0] == 1:
            #plt.imshow(np.squeeze(imgs[i]), cmap='gray')
            #plt.title(label=text_labels[int(labels[i])])
            ax.imshow(np.squeeze(imgs[i]), cmap = 'gray')
            ax.set_title(text_labels[int(labels[i])])
        else:
            # plt.imshow(np.squeeze(imgs[i]))
            # plt.title(label=text_labels[int(labels[i])])
            ax.imshow(np.squeeze(imgs[i]))
            ax.set_title(text_labels[int(labels[i])])
    plt.show()