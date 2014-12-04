from __future__ import division
from __future__ import print_function

import site
import glob
import h5py
import numpy as np
import time
import site
site.addsitedir('./cython')
import fastbin as fb
import requests
#
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def fetchurl(urlname,outname):
    """
       Download  file from its http address using the requests module
       
       Input: urlname -- string like 'http://clouds.eos.ubc.ca/~phil/Downloads/a301/A2006303_subset.h5'
              outname -- name of output file to write --  data.h5
       Returns: None
    """
    with open(outname, 'wb') as handle:
        response = requests.get(urlname, stream=True)
        if not response.ok:
            raise Exception('trouble with url')
        count=0
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
            count+=1
            if count % 10000 == 0:
                print("downloaded {:d} Mbytes".format(int(count/1000)))
    return None


def reproj_slow(raw_data, raw_x, raw_y, xlim, ylim, res):
    
    '''
    =========================================================================================
    Reproject MODIS L1B file to a regular grid
    -----------------------------------------------------------------------------------------
    d_array, x_array, y_array, bin_count = reproj_slow(raw_data, raw_x, raw_y, xlim, ylim, res)
    -----------------------------------------------------------------------------------------
    Input:
            raw_data: L1B data, N*M 2-D array.
            raw_x: longitude info. N*M 2-D array.
            raw_y: latitude info. N*M 2-D array.
            xlim: range of longitude, a list.
            ylim: range of latitude, a list.
            res: resolution, single value.
    Output:
            d_array: L1B reprojected data.
            x_array: reprojected longitude.
            y_array: reprojected latitude.
            bin_count: how many raw data point included in a reprojected grid.
    Note:
            function do not performs well if "res" is larger than the resolution of input data.
            size of "raw_data", "raw_x", "raw_y" must agree.
    =========================================================================================
    '''
    
    x_bins=np.arange(xlim[0], xlim[1], res)
    y_bins=np.arange(ylim[0], ylim[1], res)
    x_indices=np.searchsorted(x_bins, raw_x.flat, 'right')
    y_indices=np.searchsorted(y_bins, raw_y.flat, 'right')
        
    y_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    x_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    d_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    bin_count=np.zeros([len(y_bins), len(x_bins)], dtype=np.int)
    
    for n in range(len(y_indices)): #indices
        bin_row=y_indices[n]-1 # '-1' is because we call 'right' in np.searchsorted.
        bin_col=x_indices[n]-1
        bin_count[bin_row, bin_col] += 1
        x_array[bin_row, bin_col] += raw_x.flat[n]
        y_array[bin_row, bin_col] += raw_y.flat[n]
        d_array[bin_row, bin_col] += raw_data.flat[n]
                   
    for i in range(x_array.shape[0]):
        for j in range(x_array.shape[1]):
            if bin_count[i, j] > 0:
                x_array[i, j]=x_array[i, j]/bin_count[i, j]
                y_array[i, j]=y_array[i, j]/bin_count[i, j]
                d_array[i, j]=d_array[i, j]/bin_count[i, j] 
            else:
                d_array[i, j]=np.nan
                x_array[i, j]=np.nan
                y_array[i,j]=np.nan
                
    return d_array, x_array, y_array, bin_count


def reproj_fast(raw_data, raw_x, raw_y, xlim, ylim, res):
    
    '''
    =========================================================================================
    Reproject MODIS L1B file to a regular grid
    -----------------------------------------------------------------------------------------
    d_array, x_array, y_array, bin_count = reproj_fast(raw_data, raw_x, raw_y, xlim, ylim, res)
    -----------------------------------------------------------------------------------------
    Input:
            raw_data: L1B data, N*M 2-D array.
            raw_x: longitude info. N*M 2-D array.
            raw_y: latitude info. N*M 2-D array.
            xlim: range of longitude, a list.
            ylim: range of latitude, a list.
            res: resolution, single value.
    Output:
            d_array: L1B reprojected data.
            x_array: reprojected longitude.
            y_array: reprojected latitude.
            bin_count: how many raw data point included in a reprojected grid.
    Note:
            function do not performs well if "res" is larger than the resolution of input data.
            size of "raw_data", "raw_x", "raw_y" must agree.
    =========================================================================================
    '''
    minlon,maxlon=xlim
    num_xbins=int((maxlon-minlon)/res)
    minlat,maxlat=ylim
    num_ybins=int((maxlat-minlat)/res)
    lon_dict=fb.do_bins(raw_x,num_xbins,minlon,maxlon)
    lat_dict=fb.do_bins(raw_y,num_ybins,minlat,maxlat)
    y_array,x_array,d_array,bin_count=fb.hist_latlon(lon_dict,lat_dict,raw_data)
    return d_array, x_array, y_array, bin_count


if __name__=="__main__":
    h5_filename='data.h5'
    subset =glob.glob(h5_filename)[0]

    maxdim=None
    xslice=slice(0,maxdim)
    yslice=slice(0,maxdim)
    with h5py.File(subset) as h5_file:
        chan1=h5_file['channel1'][xslice,yslice]
        chan31=h5_file['channel31'][xslice,yslice]
        lats=h5_file['lattitude'][xslice,yslice]
        lons=h5_file['longitude'][xslice,yslice]

    res=0.05
    xlim=[np.min(lons), np.max(lons)]
    ylim=[np.min(lats), np.max(lats)]
    t0=time.clock()
    c31_grid, longitude, latitude, bin_count = reproj_slow(chan31, lons, lats, xlim, ylim, res)
    t1=time.clock()
    slow_time=t1-t0
    print('slow elapsed time = {:8.5f}'.format(slow_time))
    t0=time.clock()      
    c31_grid, longitude, latitude, bin_count = reproj_fast(chan31, lons, lats, xlim, ylim, res)
    t1=time.clock()
    fast_time=t1-t0
    print('fast elapsed time = {:8.5f}'.format(fast_time))
    print('speedup = {:8.1f}'.format((slow_time/fast_time)))
    plt.close('all')
    fig,ax=plt.subplots(1,1)
    ax.hist(lons.flat)
    ax.set_title('step 1')
    fig.canvas.draw()
    fig.savefig('step1.png')


    fig,ax=plt.subplots(1,1)
    ax.hist(c31_grid[~np.isnan(c31_grid)])
    ax.set_title('step 2')
    fig.canvas.draw()
    fig.savefig('step2.png')


    from mpl_toolkits.basemap import Basemap
    fig,ax=plt.subplots(1,1,figsize=(12,12))
    lcc_values=dict(resolution='l',projection='lcc',
                    lat_1=30,lat_2=50,lat_0=45,lon_0=-135,
                    llcrnrlon=-150,llcrnrlat=30,
                    urcrnrlon=-120,urcrnrlat=50,ax=ax)
    proj=Basemap(**lcc_values)
    # create figure, add axes
    ## define parallels and meridians to draw.
    parallels=np.arange(-90, 90, 5)
    meridians=np.arange(0, 360, 5)
    proj.drawparallels(parallels, labels=[1, 0, 0, 0],fontsize=10, latmax=90)
    proj.drawmeridians(meridians, labels=[0, 0, 0, 1],fontsize=10, latmax=90)
    # draw coast & fill continents
    #map.fillcontinents(color=[0.25, 0.25, 0.25], lake_color=None) # coral
    out=proj.drawcoastlines(linewidth=1.5, linestyle='solid', color='k')
    x, y=proj(longitude, latitude)
    CS=proj.pcolor(x, y, c31_grid, cmap=plt.cm.hot)
    ax.set_title('step 4')
    #
    # now add the radar ground track to the image
    #
    CBar=proj.colorbar(CS, 'right', size='5%', pad='5%')
    CBar.set_label('Channel 31 radiance (W/m^2/micron/sr')
    fig.canvas.draw()
    fig.savefig('step3.png')

## # In[ ]:



