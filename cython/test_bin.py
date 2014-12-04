from matplotlib import pyplot as plt
plt.switch_backend('Agg') #batch
#plt.switch_backend('MacOSX') #interactive
import numpy as np
from fastbin import do_bins,hist_2d
import numpy.random as nr

from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def makeRandom(meanx=None,stdx=None,meany=None,stdy=None,rho=None,
               numpoints=4000):
    """
    return a tuple with two vectors (xvec,yvec) giving the
    coordinates of numpoints chosen from a two dimensional
    Gauassian distribution

    Parameters
    ----------

    meanx: float -- mean in x direction
    stdx:  float -- standard deviation in x direction
    meany: float -- mean in y direction
    stdy:  float -- standar deviation in y direction
    numpoints:  length of returned xvec and yvec


    Returns
    -------

    (xvec, yvec): tuple of ndarray vectors of length numpoints

    Example
    -------

    invalues={'meanx':450.,
              'stdx':50,
              'meany':-180,
              'stdy':40,
              'rho':0.8}

    chanx,chany=makeRandom(**invalues)


    """
 
    nr.seed(50)
    sigma=np.array([stdx**2., rho*stdx*stdy, rho*stdx*stdy, stdy**2.])
    sigma.shape=[2,2]
    meanvec=[meanx,meany]
    outRandom=nr.multivariate_normal(meanvec,sigma,[numpoints,])
    chan1=outRandom[:,0]
    chan2=outRandom[:,1]
    return (chan1,chan2)


if __name__=="__main__":

    #
    # first bullseye centered at (x=450,y= -180)
    #
    invalues={'meanx':450.,
              'stdx':50,
              'meany':-180,
              'stdy':40,
              'rho':0.8}


    chanx,chany=makeRandom(**invalues)

    #
    # second bullseye centered at (x=50,y=-80)
    #
    bullseye={'meanx':50.,
              'stdx':14,
              'meany':-80,
              'stdy':14,
              'rho':0.0}

    chanxB,chanyB=makeRandom(**bullseye)
    chanx=np.concatenate((chanx,chanxB))
    chany=np.concatenate((chany,chanyB))


    bin_x=do_bins(chanx,70,0,700,-999,-888)
    bin_y=do_bins(chany,50,-400,0,-999,-888)



    fig1=plt.figure(1)
    fig1.clf()
    x_data=bin_x['data_vec']
    y_data=bin_y['data_vec']
    axis2=fig1.add_subplot(111)
    axis2.plot(x_data,y_data,'b.')
    axis2.set_title('scatterplot')
    fig1.canvas.draw()
    fig1.savefig('scatter.png')
    #plt.show()

    x_centers=bin_x['bin_centers']
    y_centers=bin_y['bin_centers']
    the_hist=hist_2d(bin_x,bin_y)
    counts=the_hist['count_grid']
    cmap=cm.RdBu_r
    cmap.set_over('y')
    cmap.set_under('w')
    vmin= 0.
    vmax= 300.
    the_norm=Normalize(vmin=vmin,vmax=vmax,clip=False)

    fig3=plt.figure(1)
    fig3.clf()
    axis1=fig3.add_subplot(111)
    im=axis1.pcolormesh(x_centers,y_centers,counts,cmap=cmap,norm=the_norm)
    cb=fig3.colorbar(im,extend='both')
    axis1.set_title('2d histogram')
    fig3.canvas.draw()
    fig3.savefig('histogram.png')
    
