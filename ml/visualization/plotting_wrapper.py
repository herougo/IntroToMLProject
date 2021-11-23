''' 
This is an alternate interface for matplotlib to display plots with one line of python
For example, the following line plots the 3 acceleration components in the same graph
df = fc.obdToDataframe("obd_file.csv")
acc_matrix = dfToAccMatrix(df)[:,1:]
signals(acc_matrix)
You can also pass Pandas Series (type of a DataFrame column) into some of the functions
and they will automatically format them. For example, the following code is equivalent
to the 'signals' line.
lines(df['normal_acc_mobilex'], df['normal_acc_mobiley'], df['normal_acc_mobilez'])
'''

''' Notes
plt.cla() clears an axes, i.e. the currently active axes in the current figure. It leaves the other axes untouched.
plt.clf() clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused
for other plots.
plt.close() closes a window, which will be the current window, if not specified otherwise.

'''


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D

PLT_COLOURS = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k', 
    'orange', 'lime', 'indigo', 'brown']
AUTO_SHOW = False

# Purpose: if a panda series is passed, format it to be compatible
#          with matplotlib
def format_array(x):
    if type(x) == pd.Series:
        x = x.dropna().values
    return x

def format_dict(d):
    for k in d.keys():
        d[k] = format_array(d[k])
    return d

def set_text(ax, x_label="", y_label="", title=""):
    if title != "":
        ax.set_title(title)
    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)

def histogram(ax, values, n_bins=None, x_label="", y_label="", title="", **kwargs):
    # histogram input specification:
    #  - values: list of numbers
    #  - n_bins: integer
    #  - x_label, y_label, and title: string
    if ax is None:
        ax = plt.axes()

    values = format_array(values)
    # *** TODO: flatten and filter out nan's implicitly
    result = ax.hist(values, bins=n_bins, **kwargs)
    if AUTO_SHOW:
        plt.show()

    return result

def boxplot(ax, data, x_label="", y_label="", title="", **kwargs):
    # box and whisker plot
    if ax is None:
        ax = plt.axes()

    assert isinstance(data, dict)

    index = list(range(1, 1 + len(data.keys())))
    result = ax.boxplot(list(data.values()), **kwargs)
    ax.set_xticks(index, data.keys())

    if title != "":
        ax.set_title(title)
    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)
    if AUTO_SHOW:
        plt.show()

    return result

def scatter(ax, x_values, y_values, labels=[], x_label="", y_label="", title="", **kwargs):
    # scatter plot input specification:
    #  - x_values, y_values: list of numbers (same size)
    #  - labels: same size as x_values or an empty list
    #  - x_label, y_label, and title: string
    if ax is None:
        ax = plt.axes()

    x_values = format_array(x_values)
    y_values = format_array(y_values)

    if len(labels) == 0:
        scatter_handle = ax.scatter(x_values, y_values, **kwargs)
        handles = [scatter_handle]
    else:
        distinct_labels = set(labels)
    
        if len(distinct_labels) > len(PLT_COLOURS):
            raise Exception("Too many labels: {} > {}".format(len(distinct_labels), len(PLT_COLOURS)))
        
        # group by label
        label_dict = dict([(key, []) for key in distinct_labels])
        for x, y, label in zip(x_values, y_values, labels):
            label_dict[label].append((x, y))
            
        # iterate by label
        label_to_index = dict([(label, i) for i, label in enumerate(distinct_labels)])
        handles = []
        for label in label_dict.keys():
            values = label_dict[label]
            label_x_values, label_y_values = zip(*values)
            #label_x_values, label_y_values
            colour = PLT_COLOURS[label_to_index[label]]
            
            scatter_handle = ax.scatter(label_x_values, label_y_values, label=label, 
                    c=colour, cmap=matplotlib.colors.ListedColormap(PLT_COLOURS), **kwargs)
            handles.append(scatter_handle)
        
        ax.legend(handles=handles)

    if title != "":
        ax.set_title(title)
    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)
    
    if AUTO_SHOW:
        plt.show()

    return handles

def scatter3d(x_values, y_values, z_values, labels=[], x_label="x", y_label="y", z_label="z", title=""):
    # 3d scatter plot input specification:
    #  - x, y, z: list of numbers (same size)
    #  - labels: same size as x_values or an empty list
    #  - x_label, y_label, z_label, and title: string
    x_values = format_array(x_values)
    y_values = format_array(y_values)
    z_values = format_array(z_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if len(labels) == 0:
        scatter_handle = ax.scatter(x_values, y_values, z_values, label=labels)
        handles = [scatter_handle]
    else:
        distinct_labels = set(labels)
    
        if len(distinct_labels) > len(PLT_COLOURS):
            raise Exception("Too many labels: {} > {}".format(len(distinct_labels), len(PLT_COLOURS)))
        
        # group by label
        label_dict = dict([(key, []) for key in distinct_labels])
        for x, y, z, label in zip(x_values, y_values, z_values, labels):
            label_dict[label].append((x, y, z))
            
        # iterate by label
        label_to_index = dict([(label, i) for i, label in enumerate(distinct_labels)])
        handles = []
        for label in label_dict.keys():
            values = label_dict[label]
            label_x_values, label_y_values, label_z_values = zip(*values)
            #label_x_values, label_y_values
            colour = PLT_COLOURS[label_to_index[label]]
            
            scatter_handle = ax.scatter(label_x_values, label_y_values, label_z_values, label=label, 
                    c=colour, cmap=matplotlib.colors.ListedColormap(PLT_COLOURS))
            handles.append(scatter_handle)
        
        ax.legend(handles=handles)
    
    # default values (access with ax.axes.azim for example):
    #  - azimuth:   azim = -60
    #  - elevation: elev = 30
    #  - distance:  dist = 10
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    if title != "":
        ax.set_title(title)
    ax.view_init(elev=30, azim=-60)
    #for i in xrange(0,360,30):
        #ax.view_init(elev=10., azim=i)
        #plt.savefig("plot-movie/movie%d.png" % i)
    if AUTO_SHOW:
        plt.show()

    return handles


def line(ax, x_values, y_values, x_label="", y_label="", title="", **kwargs):
    # line graph input specification:
    #  - x_values and y_values: list of numbers (same size)
    #  - x_label, y_label, and title: string
    if ax is None:
        ax = plt.axes()

    x_values = format_array(x_values)
    y_values = format_array(y_values)
    result = ax.plot(x_values, y_values, **kwargs)
    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)
    if title != "":
        ax.set_title(title)
    if AUTO_SHOW:
        plt.show()

    return result

def save(file_name):
    plt.savefig(file_name)

def save_and_clear(file_name):
    plt.savefig(file_name)
    plt.clf()

def clear():
    # clear figure
    plt.clf()

def bar(ax, data, x_label="", y_label="", title="", **kwargs):
    # bar graph input specification:
    #  - data: dictionary
    #     - keys are possible categories
    #     - values are the number of instances of the category corresponding to the key
    #    or list of labels
    #     - for each index i, the category is i and the number of the category is data[i]
    if ax is None:
        ax = plt.axes()

    bar_width = 0.7

    if type(data) == dict:
        index = np.arange(len(data.keys()))
        rects1 = ax.bar(index + bar_width / 2, data.values(), bar_width, **kwargs)
        ax.set_xticks(index + bar_width, data.keys())
    else:
        index = np.arange(len(data))
        rects1 = ax.bar(index + bar_width / 2, data, bar_width, **kwargs)
        ax._set_xticks(index + bar_width, index)
    
    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)
    if title != "":
        ax.set_title(title)
    ax.xaxis.set_ticks_position('none')
    
    if AUTO_SHOW:
        plt.show()

    return rects1


def confusion(data, labels, annotate=False):
    # confusion matrix plot input specification:
    #  - data: NxN matrix
    #  - labels: size N list of corresponding labels to the data (labels are plotted from
    #    left to right and up to down)
    data = np.array(data)
    
    im = plt.imshow(data, cmap='Reds', interpolation='nearest')
    plt.colorbar(im, orientation='vertical')
    #plt.grid(True)
    #plt.rc('grid', linestyle="-", color='black')
    
    # setting labels
    index = np.arange(len(labels))
    ax = plt.axes()
    ax.set_xticks(index, labels)
    ax.set_yticks(index, labels)
    
    # hide ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # annotations
    if annotate:
        width, height = data.shape
        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(data[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
    
    if AUTO_SHOW:
        plt.show()

    return im


def lines(ax, *lines, **kwargs):
    # multiple lines plot input specification
    #  - lines is either a dictionary or an arbitrary number of lists of numbers passed to the function
    if ax is None:
        ax = plt.axes()

    handles = []

    if len(lines) == 1 and type(lines[0]) == dict:
        for k in lines[0]:
            line = format_array(lines[0][k])
            line_handle, = ax.plot(line, label=str(k), **kwargs)
            handles.append(line_handle)
    else:
        counter = 0
        
        for line in lines:
            line = format_array(line)
            line_handle, = ax.plot(line, label=str(counter), **kwargs)
            handles.append(line_handle)
            counter += 1
    
    ax.legend(handles=handles)
    if 'x_label' in kwargs.keys():
        ax.set_xlabel(kwargs['x_label'])
    if 'y_label' in kwargs.keys():
        ax.set_ylabel(kwargs['y_label'])
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    if AUTO_SHOW:
        plt.show()

    return (line_handle,)


def signals(ax, mat):
    # Purpose: plot the columns of 2d matrix mat as lines on the same graph
    if ax is None:
        ax = plt.axes()

    lines(ax, *np.array(mat).T)

def isocontours_lines(ax, func, xlimits=[-3, 3], ylimits=[-3, 3], numticks=101, 
                      pytorch_func=False, **kwargs):
    # func: mapping Nx2 to scalar (ie x, y coordinates)
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    func_input = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    if pytorch_func:
        func_input = torch.tensor(func_input)
    zs = func(func_input)
    if isinstance(zs, torch.Tensor):
        zs = zs.detach().numpy()
    Z = zs.reshape(X.shape)
    plot = plt.contour(X, Y, Z, **kwargs)
    #ax.set_yticks([])
    #ax.set_xticks([])
    return plot

def isocontours(ax, func, xlimits=[-3, 3], ylimits=[-3, 3], numticks=101, pytorch_func=False):
    # func: mapping Nx2 to scalar (ie x, y coordinates)
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    func_input = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    if pytorch_func:
        func_input = torch.tensor(func_input)
    zs = func(func_input)
    if isinstance(zs, torch.Tensor):
        zs = zs.detach().numpy()
    Z = zs.reshape(X.shape)
    plot = plt.contourf(X, Y, Z)
    #ax.set_yticks([])
    #ax.set_xticks([])
    return plot


if __name__ == "__main__":
    #AUTO_SHOW = True
    time = [0, 1, 2]
    mobilex = [0, 1, 2]
    mobiley = [0, 1, 2]
    ax = plt.axes()

    mydict = {"A": 20, "B": 35, "C": 30, "D": 35, "E": 27}
    bar(ax, mydict)
    save_and_clear('hi.jpeg')

    scatter(None, mobilex, mobiley)
    save_and_clear('hi3.jpeg')


    z=np.array(((21,1,0),
                (0,25,6),
                (0,0,22)))

    labels = ["a", "b", "c"]
    confusion(z, labels)
    save('hi2.jpeg')
