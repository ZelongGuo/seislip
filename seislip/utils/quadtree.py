"""
Quadtree down-sampling for InSAR images.

Author: Zelong Guo
Email: zelong.guo@outlook.com
Create on 27, Dec. 2023

"""

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colorbar as cbar

import math
import numpy as np

__author__ = "Zelong Guo"

class Node(object):
    """Node class to generate a node (parent node or child node) which start
    with pixel coordinate/index (x0, y0) with certain width (width) and height (height)."""
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

        self.children = []

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points(self, image):
        return image[self.x0:self.x0 + self.get_width(), self.y0:self.y0 + self.get_height()]

    def get_std(self, image):
        pixels = self.get_points(image)
        # get the standard deviation of non-zero elements of the image
        nonzero_count = np.count_nonzero(pixels)
        if nonzero_count == 0:
            std = 0
        else:
            pixels = pixels[pixels != 0]
            std = np.std(pixels)
        return std


class QTree(object):
    """Quadtree class.

    Args:
        - x:          2-D mesh of longitudes or UTM X
        - y:          2-D mesh of latitudes or UTM Y
        - image:      2-D matrix of an image

    Return:
        - None.
    """
    def __init__(self, x, y, image):
        self.x, self.y = x, y
        self._replace_nan_to_zero(image)

        self.root = Node(0, 0, image.shape[0], image.shape[1])
        self.qtscatter = None
        self.qtrect = None
        self.qtnumber = None

    def _replace_nan_to_zero(self, image):
        self.image = np.where(np.isnan(image), 0, image)


    def subdivide(self, mindim, maxdim, std_threshold):
        """Do quadtree to get the quadtree down-sampling results.

        Args:
            - mindim:            minimum pixels number as the minimum quadtree box size
            - maxdim:            maximum pixels numbers as the maximum quadtree box size
            - std_threshold:     threshold of the standard deviation

        Return:
            - None.
        """
        recursive_subdivide(node=self.root, image=self.image, mindim=mindim, maxdim=maxdim,
                            std_threshold=std_threshold)

    def qtresults(self, nonzero_fraction=0.3):
        """If the non-zero fraction in a block are smaller than nonzero_fraction,
        then we will ignore it by assigning it with nan.
        """
        c = find_children(self.root)
        xyz = []
        xywhz = []
        xy4GMT = []
        z4GMT = []
        for n in c:
            pixels = n.get_points(self.image)
            x = n.get_points(self.x)
            y = n.get_points(self.y)
            new_x, new_y = x.mean(), y.mean()
            nonzero_count = np.count_nonzero(pixels)
            if nonzero_count / (n.width * n.height) > nonzero_fraction:
                nonzero = pixels[pixels != 0]
                mean_value = np.mean(nonzero)
                xyz.append([new_x, new_y, mean_value])
                # xywhz.append([x[0, 0], y[0, 0], n.width, n.height, mean_value])
                xywhz.append([x[-1, 0], y[-1, 0], abs(x[-1, -1]-x[-1, 0]), abs(y[0, 0]-y[-1, 0]), mean_value])
                xy4GMT.append([
                    [x[-1, 0], y[-1, 0]],
                    [x[-1, -1], y[-1, -1]],
                    [x[0, -1], y[0, -1]],
                    [x[0, 0], y[0, 0]]
                ])
                z4GMT.append([mean_value])

        self.qtscatter = np.array(xyz)
        self.qtrect = np.array(xywhz)
        self.qtnumber = self.qtrect.shape[0]
        self.qtxy4GMT = np.array(xy4GMT)
        self.qtz4GMT = np.array(z4GMT)
        print("+-" * 30)
        print(f"Quadtree downsampling: the number of segments are {self.qtnumber}")
        print("+-" * 30)


    def show_qtresults(self, figtitle = None, xlabel = None, ylabel = None,
                       clabel = None, save_as_file = "no", filename = None):
        plt.figure(figsize=(15, 3.5))
        plt.subplot(1, 3, 1)
        plt.title(f"{figtitle}")
        vmin, vmax = self.image.min(), self.image.max()
        img = np.where(self.image == 0, np.nan, self.image)
        plt.imshow(img, cmap="rainbow", extent=[np.min(self.x), np.max(self.x), np.min(self.y),
                                                np.max(self.y)], vmin=vmin, vmax=vmax, origin="upper")
        # plt.scatter(self.x.reshape(-1, 1), self.y.reshape(-1, 1), c=img.reshape(-1, 1),
        #             cmap="rainbow", vmin=vmin, vmax=vmax)
        plt.xlim([np.min(self.x), np.max(self.x)])
        plt.ylim([np.min(self.y), np.max(self.y)])
        if xlabel is not None and ylabel is not None:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        plt.colorbar(label=f"{clabel}")

        plt.subplot(1, 3, 2)
        plt.title(f"Scatters ({self.qtnumber} points)")
        plt.scatter(self.qtscatter[:, 0], self.qtscatter[:, 1], c=self.qtscatter[:, 2], cmap="rainbow",
                    vmin=vmin, vmax=vmax)
        plt.xlim([np.min(self.x), np.max(self.x)])
        plt.ylim([np.min(self.y), np.max(self.y)])
        plt.colorbar()

        ax = plt.subplot(1, 3, 3)
        plt.title(f"Rectangle Patches ({self.qtnumber} points)")
        normal = plt.Normalize(self.image.min(), self.image.max())
        for i in range(self.qtrect.shape[0]):
            color = plt.cm.rainbow(normal(self.qtrect[i, 4]))
            rect = patches.Rectangle((self.qtrect[i, 0], self.qtrect[i, 1]), self.qtrect[i, 2],
                                     self.qtrect[i, 3], facecolor=color, edgecolor="black", fill=True, linewidth=0.5)
            ax.add_patch(rect)
        cax, _ = cbar.make_axes(ax)
        cb2 = cbar.ColorbarBase(cax, cmap=plt.cm.rainbow, norm=normal)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_ylim(self.y.min(), self.y.max())
        if save_as_file != "no":
            plt.savefig(filename, dpi=300)
        else:
            plt.show()
        plt.close()


def _should_split(node, image, mindim, maxdim, ste_threshold):
    if node.width > maxdim or node.height > maxdim:
        split = 1
    elif node.width <= mindim or node.height <= mindim:
        split = 0
    else:
        if node.get_std(image) > ste_threshold:
            split = 1
        else:
            split = 0
    return split

def recursive_subdivide(node, image, mindim, maxdim, std_threshold):
    split = _should_split(node, image, mindim, maxdim, std_threshold)

    if split == 1:
        middle_w1, middle_h1 = math.floor(node.width / 2), math.floor(node.height / 2)
        middle_w2, middle_h2 = math.ceil(node.width / 2), math.ceil(node.height / 2)

        x1 = Node(node.x0, node.y0, middle_w1, middle_h1)  # top left
        recursive_subdivide(x1, mindim=mindim, maxdim=maxdim, std_threshold=std_threshold, image=image)
        x2 = Node(node.x0 + middle_w1, node.y0, middle_w2, middle_h1)  # topo right
        recursive_subdivide(x2, mindim=mindim, maxdim=maxdim, std_threshold=std_threshold, image=image)
        x3 = Node(node.x0, node.y0 + middle_h1, middle_w1, middle_h2)  # btm left
        recursive_subdivide(x3, mindim=mindim, maxdim=maxdim, std_threshold=std_threshold, image=image)
        x4 = Node(node.x0 + middle_w1, node.y0 + middle_h1, middle_w2, middle_h2) # btm right
        recursive_subdivide(x4, mindim=mindim, maxdim=maxdim, std_threshold=std_threshold, image=image)
        node.children = [x1, x2, x3, x4]

# def recursive_subdivide(node, image, mindim, maxdim, std_threshold):

    # if node.get_std(image) <= std_threshold:
    #     return
    # middle_w1, middle_h1 = math.floor(node.width / 2), math.floor(node.height / 2)
    # middle_w2, middle_h2 = math.ceil(node.width / 2), math.ceil(node.height / 2)
    #
    # if middle_w1 <= mindim or middle_h1 <= mindim:
    #     return
    #
    # x1 = Node(node.x0, node.y0, middle_w1, middle_h1)  # top left
    # recursive_subdivide(x1, mindim=mindim, std_threshold=std_threshold, image=image)
    # x2 = Node(node.x0 + middle_w1, node.y0, middle_w2, middle_h1)  # topo right
    # recursive_subdivide(x2, mindim=mindim, std_threshold=std_threshold, image=image)
    # x3 = Node(node.x0, node.y0 + middle_h1, middle_w1, middle_h2)  # btm left
    # recursive_subdivide(x3, mindim=mindim, std_threshold=std_threshold, image=image)
    # x4 = Node(node.x0 + middle_w1, node.y0 + middle_h1, middle_w2, middle_h2) # btm right
    # recursive_subdivide(x4, mindim=mindim, std_threshold=std_threshold, image=image)
    # node.children = [x1, x2, x3, x4]
   

def find_children(node):
   if not node.children:
       return [node]
   else:
       children = []
       for child in node.children:
           children += (find_children(child))
   return children


# TODO: write to files as scatter and rectangles patches which are GMT needed.



if __name__ == '__main__':

    delta = 0.025
    x = y = np.arange(-3.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 20
    img = Z
    # # set up nan
    # img[:, 0:120] = 0
    # img[:, -1] = 0
    # img[0, :] = 0
    # img[-30:-1, :] = 0


    print("=-" * 30)
    qtTemp = QTree(X, Y, img)  #contrast threshold, min cell size, img
    qtTemp.subdivide(4, 64, np.std(img)-2) # recursively generates quad tree
    qtTemp.qtresults(0.3)
    qtTemp.show_qtresults("orignal data", 'none')
