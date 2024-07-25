# load libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy
from scipy.io import wavfile
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np

import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

# Import the necessaries libraries
#import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
#pyo.init_notebook_mode()

from sklearn.metrics import pairwise
import plotly.graph_objs as go
from ipywidgets import widgets, interactive_output, interact
from matplotlib import animation
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm
from sklearn.preprocessing import minmax_scale


# Functions


# audio preprocessingn and play
# --------------------------------------------   
def wavPlayer(filepath):
    """ 
    # this is a wrapper that take a filename and publish an html <audio> tag to listen to it
    will display html 5 player for compatible browser

    Parameters :
    ------------
    filepath : relative filepath with respect to the notebook directory ( where the .ipynb are not cwd)
               of the file to play

    The browser need to know how to play wav through html5.

    there is no autoplay to prevent file playing when the browser opens
    """

    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>

    <body>
    <audio controls="controls" style="width:600px" >
      <source src="files/%s" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """%(filepath)
    display(HTML(src))


# data processing
#---------------------------------------------
def  get_chunked_data(x1,x2,kernel,slide):

    '''
    returns chunks of data for a given kernal and window size
    '''
    noof_chunks = int(len(x1)/slide)-2

    x1_chunked = [x1[slide*i: slide*i+kernel] for i in range(noof_chunks)]
    x2_chunked = [x2[slide*i: slide*i+kernel] for i in range(noof_chunks)]

    # make a data frame with this data
    df = pd.DataFrame()
    df['x1'] = x1_chunked
    df['x2'] = x2_chunked

    return df


# visualization
# --------------------------------------------   

class parameter:
    '''wrapper class
    to use this you just need to define a function and that takes in the parameters as input
    and also a detailed lsit of parameters [description,value, min, max, step]
    '''


    def __init__(self, description,value, min_val, max_val, step):

        self.description = description     # name of parameter
        self.value = value                 # inital value of parameter
        self.min = min_val
        self.max = max_val
        self.step = step

def widget_dynamics(function,parameters, fixed_params=None):
    '''for a given function
       note: parameter list has to be
    '''
    print("navy")
    # make a lsit of  to save a widget for each parameter
    widgets_list = []
    for para in parameters:

        widgets_list.append(widgets.FloatSlider(value=para.value,
                                                 min=para.min,
                                                 max = para.max,
                                                 step= para.step,
                                                 description=para.description,
                                                 orientation='vertical',
                                                 #layout=widgets.Layout(width='30%',height='30%',),
                                                 style={'bar_color': '#ffff00'}
                                                ))

    # make a spererate dictionary to save the widgets
    widget_dictionary = {}
    for widget in widgets_list:
        widget_dictionary[widget.description]=widget

    vec = widgets.HBox(widgets_list)

    if fixed_params is not None:
        def wrapped_function(**kwargs):
            return function(new_param=fixed_params, **kwargs)
    else:
        def wrapped_function(**kwargs):
            return function(**kwargs)

    w = interactive_output(wrapped_function,widget_dictionary)

    # display widgets
    display(vec,w)

    '''
     ** **************************************example*******************************
    # define your function
    def my_function(omega,beta):
        print(omega,beta)

    ## define your parameters
    my_parameters = [parameter("A",2.0,0.01,100,0.01),
                      parameter("omega",2.0,0.01,100,0.01)]

    ## call widgets
    widget_dynamics(function=plot_dynamical_system,
                   parameters=parameters)
    '''

def plot_scatter(x):
    fig = go.Figure(data=go.Scattergl(
                                x = [i for i in range(x.shape[0])],
                                y = x,
                                text=[i for i in range(x.shape[0])],
                                mode ="markers",
                                marker=dict(size=3,color='yellow',opacity=0.5)
                                ))                                   #color=[i for i in range(x_plot.shape[0])]

    fig.update_layout(template="plotly_dark", width=1200, height=700)
    #fig.update_xaxes(rangeslider_visible=True)
    fig.show()

def plot_recursion(index,size,tau):
    '''
    function to plot recursion for time series for given index
    tau: dealy time
    '''
    index = np.int32(index)
    size = np.int32(size)
    tau = np.int32(tau)

    X = timeseries_data[index:index+size]
    X = np.array(X).reshape(-1,1)

    plt.figure(figsize=(30,10))
    #plt.style.use("seaborn-dark")
    for param in ['text.color', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = 'black'  # bluish dark grey
    plt.rcParams['text.color'] = "white"

    plt.subplot(1,3,1)
    plt.scatter(x = [i for i in range(X.shape[0])],
                y = X,s=4.0,c='gold')
    plt.subplot(1,3,2)
    plt.imshow(pairwise.pairwise_distances(X),"gray")
    #plt.colorbar()

    # delay cordinates
    x1,x2 = get_delay_coordinates(X,tau)
    plt.subplot(1,3,3)
    plt.scatter(x1,x2,s=4.0,c='gold')
    plt.show()

def plot_attractor(index,size,tau,g_alpha,g_size, c_size,fixed_params):
    '''
    Plots only attractor
    '''
    timeseries_data = fixed_params 
    index = np.int32(index)
    size = np.int32(size)
    tau = np.int32(tau)

    X = timeseries_data[index:index+size]
    X = np.array(X).reshape(-1,1)
    x1,x2 = get_delay_coordinates(X,tau)
    n = len(x1)
    half = np.int32(n/3)

    plt.figure(figsize=(6,6))
    #plt.style.use("seaborn-dark")
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = 'black'  # bluish dark grey

    # glow
    for i in range(1,3):
        plt.scatter(x1[:half],x2[:half],
                    color="gold",          # glow color
                    alpha=g_alpha,             # glow alpha g_alpha  0.2
                    linewidth=g_size*i        # glow size  g_size    2.0
                   )
        plt.scatter(x1[half:2*half],x2[half:2*half],
                    color="blue",          # glow color
                    alpha=g_alpha,             # glow alpha g_alpha  0.2
                    linewidth=g_size*i        # glow size  g_size    2.0
                   )
        plt.scatter(x1[2*half:],x2[2*half:],
                   color="green",          # glow color
                   alpha=g_alpha,             # glow alpha g_alpha  0.2
                   linewidth=g_size*i        # glow size  g_size    2.0
                  )


    plt.scatter(x1,x2,
               marker="o",
               color="white",            # center color
               s=c_size)                    # center size c_size     1.5

    '''
    #plt.scatter(x1,x2,s=3.0,c='y')
    #url = 'https://image.freepik.com/free-photo/light-smoke-fragments-black-background_23-2148092646.jpg'
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))

    #min_val = 1.3*(X.min())
    #max_val = 1.3*(X.max())
    #plt.imshow(img,extent=[min_val,max_val,min_val,max_val])
    '''


    plt.show()

def plot_attractor_animation(x1,x2,crop_size,sound):

    '''
    x1,x2 delay coordinates
    crop_size: no of datapoints to plot for each frame
    '''
    # label crop frame
    no_of_crops = int(len(x1)/crop_size)
    t = [[i]*crop_size for i in range(no_of_crops)]
    crop_index = [item for sublist in t for item in sublist]

    # crop color
    n = crop_size
    crop_color_indx = ['A']*(int(n/3)) + ['B']*(int(n/3)) + ['C']*(int(n-2*n/3)+1)
    crop_color_indx = crop_color_indx[:n]
    color_indx = crop_color_indx*no_of_crops
    #for missing color values
    no_of_missing = len(crop_index)-len(color_indx)
    all_color_indx = color_indx + ['A']*no_of_missing

    # data frame
    n_data = len(crop_index) # only consider the datapoints from the crop and ignore the rest
    df = pd.DataFrame()
    df['x1'] = x1[:n_data]
    df['x2'] = x2[:n_data]
    df['crop_index'] = crop_index[:n_data]
    df['color_indx'] = all_color_indx[:n_data]

    # plot animation
    fig = px.scatter(df, x="x1", y="x2",color='color_indx',
                 animation_frame="crop_index",
                range_x=[sound.min(),sound.max()],
                range_y=[sound.min(),sound.max()])
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(template="plotly_dark")
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(showgrid=False,zeroline=False)
    fig.update_layout(width = 800,
                        height = 800,
                       )
    fig.update_layout(transition = {'duration': 100})
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000  # milli sec
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2000 # milli sec
    fig.show()

def plot_3D_delaycoordinates(x1,x2,x3,from_id, to_id):

    df = pd.DataFrame()
    df['x1']= x1
    df['x2']= x2
    df['x3']= x3

    my_clr = [0]*500 + [0.5]*500 + [0.9]*500

    #from_id = chunk_idx*chunk_size
    #to_id = from_id + 1500
    fig = go.Figure(data=[go.Scatter3d(x=x1[from_id:to_id],
                                       y=x2[from_id:to_id],
                                       z=x3[from_id:to_id],
                                       mode='markers',
                                       marker=dict(size=2,
                                                   color = 'yellow',
                                                   #colorscale='rainbow'
                                                   ))])
    #fig.update_layout(template="plotly_dark")
    fig.update_layout(width = 800,
                      height = 800,
                      template="plotly_dark"
                           )

    fig.show()

def Attractor_3D_animation(x1,x2,x3,chunk_size):

    no_of_chunks = int(len(x1)/chunk_size)
    x = [x1[chunk_size*i: chunk_size*(i+1)] for i in range(no_of_chunks)]
    y = [x2[chunk_size*i: chunk_size*(i+1)] for i in range(no_of_chunks)]
    z = [x3[chunk_size*i: chunk_size*(i+1)] for i in range(no_of_chunks)]

    # Create figure
    fig = go.Figure(
        data=[go.Scatter3d(x=[], y=[], z=[],
                         mode="markers",marker=dict(color="black", size=2))])

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }
    # slidesr
    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # frames
    frames = [go.Frame(data= [go.Scatter3d(
                                           x=x[k],
                                           y=y[k],
                                           z=z[k])],

                       traces= [0],
                       name=f'frame{k}'
                      )for k  in  range(len(x))]
    fig.update(frames=frames),
    fig.update_layout(updatemenus = [
                {"buttons": [
                        {
                            "args": [None, frame_args(24)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
                     sliders=sliders)
    fig.show()

def save_2D_attractor_gif(x1,x2,kernel,slide, name_gif):

    # get data chunks
    noof_chunks = int(len(x1)/slide)-2
    df = get_chunked_data(x1,x2,kernel,slide)

        # Animation with multicolor and glow
    fig,ax = plt.subplots(figsize=(10,10))
    plt.style.use("seaborn-dark")

    # function to update plots for each frame
    def update_scatterPlot(i):

        ax.clear()
        colors = [
                '#08F7FE',  # teal/cyan
                '#FE53BB',  # pink
                '#F5D300',  # yellow
                '#00ff41', # matrix green
                ]
        #seperate data for color
        n = len(df['x1'][i])
        half = np.int32(n/3)

        # circle
        for id in range(1,4):
            ax.scatter(df['x1'][i][:half],df['x2'][i][:half],
                       c="gold",   # glow color
                       alpha=0.07,     # glow alpha
                       linewidths=1.3*id)   # glow size
            ax.scatter(df['x1'][i][half:2*half],df['x2'][i][half:2*half],
                       c="blue",   # glow color
                       alpha=0.07,     # glow alpha
                       linewidths=1.3*id)   # glow size
            ax.scatter(df['x1'][i][2*half:],df['x2'][i][2*half:],
                       c="green",   # glow color
                       alpha=0.07,     # glow alpha
                       linewidths=1.3*id)   # glow size

        ax.scatter(df['x1'][i],df['x2'][i],
                   marker='o',
                   c='white',        # center color
                   s=0.6)            # center size


        ax.set_facecolor('black')
        #ax.set_xlim([min(df['x1'][i]), max(df['x1'][i])])
        #ax.set_ylim([min(df['x1'][i]), max(df['x1'][i])])
        ax.set_xlim([min(x1), max(x2)])
        ax.set_ylim([min(x1), max(x2)])
    # main animation function
    anim = animation.FuncAnimation(fig,update_scatterPlot,frames = noof_chunks,interval = 180)

    #anim.save(name_gif,writer='pillow')
    return anim

# Embedding
# -------------------------------------------- 
def get_delay_coordinates(X,tau):
    '''
    X: time series data
    tau: delay time
    '''
    x1 = [X[i] for i in range(len(X)-tau)]
    x2 = [X[i+tau] for i in range(len(X)-tau)]

    return x1,x2

def get_3D_delay_coordinates(X,tau,D):
    '''
    X: time series data
    tau: delay time
    D: dimension
    '''
    x1 = [X[i] for i in range(len(X)-D*tau)]
    x2 = [X[i+tau] for i in range(len(X)-D*tau)]
    x3 = [X[i+D*tau] for i in range(len(X)-D*tau)]

    return x1,x2,x3

def delay_coordinates_highDim(x, tau, m):
    """
    Generate delay coordinates for a time series data

    Parameters:
        x (array-like): 1-D time series data
        tau (int): delay time
        m (int): embedding dimension

    Returns:
        2-D array of shape (N - (m - 1)*tau, m), where N is the length of x
    """
    N = len(x)
    X = np.empty((N - (m - 1)*tau, m))
    for i in range(m):
        X[:, i] = x[i*tau : (i*tau + X.shape[0])]
    return X

def mutual_information(x, y, bins=50):   # source: https://jinjeon.me/post/mutual-info/
    """
    measure the mutual information of the given two images

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram

    Returns
    -------
    calculated mutual information: float

    """
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins)

    # convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal x over y
    py = np.sum(pxy, axis=0)  # marginal y over x
    px_py = px[:, None] * py[None, :]  # broadcast to multiply marginals

    # now we can do the calculation using the pxy, px_py 2D arrays
    nonzeros = pxy > 0  # filer out the zero values
    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))

def plot_mutual_information(x):
  '''
  x: time series to embed
  '''
  mi = []
  for t in range(1,100):
    delay_sol = delay_embedding(x,t,2)
    mi.append(mutual_information(delay_sol[:,0],delay_sol[:,1]))

  fig = go.Figure(data=go.Scattergl(
                              x = [i for i in range(len(mi))],
                              y = mi,
                              text=[i for i in range(len(mi))],
                              mode ="markers",
                              marker=dict(size=3,color='yellow',opacity=0.5)
                              ))                                   #color=[i for i in range(x_plot.shape[0])]

  fig.update_layout(template="plotly_dark", width=1000, height=600)
  #fig.update_xaxes(rangeslider_visible=True)
  fig.show()

def delay_embedding(time_series,τ,D):
    '''
    Returns the time delay embedding of time series of scalar data
    Args:
        time_series: scalar array
    τ =  time delay between values in phase space reconstruction
        D:  embedding dimension
    Returns:
        array of embedded vectors:
        [x[i],x[i+tau],x[i+2*tau],...,x[i + (m-1)*tau]]
    '''
    indexes = np.arange(0,D,1)*τ
    return np.array([time_series[indexes +i] for i in range(len(time_series)-(D-1)*τ)])

# extra
#---------------------------------------------
def update_voronoi(i):

    fig,ax = plt.subplots(figsize=(8,8))
    ax.clear()
    rect = ax.patch  # a rectangle instance
    rect.set_facecolor('black')
    points =np.column_stack((df['x1'][i],df['x2'][i]))
    voronoi_set = Voronoi(points)
    plt.scatter(points[:,0],points[:,1],c='y',alpha=0.2)



    voronoi_plot_2d(voronoi_set,ax=ax,line_colors='r',line_alpha=0.3,show_points=True,line_width=3.0,
                          point_size=3,show_vertices=False,ver_size=1)
    voronoi_plot_2d(voronoi_set,ax=ax,line_colors='b',line_alpha=0.5,show_points=True,line_width=1.0,
                          point_size=3,show_vertices=False,ver_size=1)
    plt.show()