import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t
cimport cython
from libc.stdio cimport printf

@cython.embedsignature(True)
@cython.cdivision(True)
@cython.wraparound(True)
@cython.boundscheck(True)
def do_bins(object data_vec,int the_numbins,float mindata,
            float maxdata,
            int missingLow= -200,int missingHigh= -100):
    """
       bin the items in data_vec into numbins
       see binit docstring for details

       parameters
       ----------

        data_vec: numpy 1d array or list
             vector of floating point numbers to be binned

        the_numbins: number of bins
        mindata:     value of leftmost edge of the bins
        maxdata:     value of rightmost edge of the bins
        missingLow   bin index assigned to data smaller than mindata
                      (must be negative)
        missingHigh  bin index assigned to data larger than maxdatat
                      (must be negative)
       returns:
       --------

         dictionary with the following keys:

         bin_count: numpy vector of length numbins (int)
            vector containing bin counts

         bin_index: numpy vector of len(data_vec)
            vector containing the bin number of each datapoint

         lowcount: int
            number of points that were smaller
            than the smallest bin

         highcount: int
            number of points that were larger than the largest
            bin

         data_vec: ndarray of the data that was binned 
    
       example
       -------
             #bin a temperature vector into 20 1K  bins
             #starting at 270.
             bin_dict=obj.do_bins(temp_vec,20,270.,300.)
    """   
    data_vec=np.ascontiguousarray(data_vec)
    data_vec=data_vec.astype(np.float64)
    cdef double* dataPtr= <double*> np.PyArray_DATA(data_vec)
    cdef double binsize=(maxdata-mindata)/float(the_numbins)
    cdef np.int32_t[:] bin_index=np.empty([data_vec.size],dtype=np.int32)
    cdef np.int32_t[:] bin_count=np.zeros([the_numbins],dtype=np.int32)
    cdef int lowcount=0, highcount=0,tot_loops=data_vec.size
    cdef int i
    cdef double float_bin,data_val
    cdef int ibin
    cdef double minval=mindata
    cdef int missingLowValue=missingLow
    cdef int missingHighValue =missingHigh
    cdef int numbins=the_numbins
    cdef double dataval

    for i in range(tot_loops):
        dataval=dataPtr[i]
        float_bin =  ((dataval - minval) /binsize)
        if float_bin < 0:
            lowcount+=1
            bin_index[i]=missingLowValue
            continue
        if float_bin >= numbins:
            highcount += 1
            bin_index[i] = missingHighValue
            continue
        ibin=<int> float_bin
        bin_count[ibin]+=1
        bin_index[i]=ibin

    bin_edges=[]
    bin_centers=[]
    bin_edges=[minval + (i*binsize) for i in range(numbins+1)]
    bin_centers=[(bin_edges[i] + bin_edges[i+1])/2. for i in range(numbins)]
    bin_edges=np.array(bin_edges)
    bin_centers=np.array(bin_centers)
    out_dict=dict(bin_count=np.asarray(bin_count),bin_index=np.asarray(bin_index),
                  bin_edges=bin_edges,bin_centers=bin_centers,numbins=the_numbins,
                  lowcount=lowcount,highcount=highcount,data_vec=data_vec)
    return out_dict


def hist_latlon(dict lon_dict, dict lat_dict,object rads):

    lon_index=lon_dict['bin_index']
    lat_index=lat_dict['bin_index']
    lats=lat_dict['data_vec'].ravel()
    lons=lon_dict['data_vec'].ravel()
    cdef int numlatbins=lat_dict['numbins']
    cdef int numlonbins=lon_dict['numbins']
    out_vals=np.empty([numlatbins,numlonbins],dtype=np.object)

    cdef int num_datapts=<int>lat_index.size
    cdef int row,col, data_index, grid_row,grid_col,rows,cols


    cdef int[:] lat_view=lat_index
    cdef int[:] lon_view=lon_index
    
    for row in range(numlatbins):
        for col in range(numlonbins):
            out_vals[row,col]=list()

    for data_index in range(num_datapts):
        grid_row=lat_index[data_index]
        grid_col=lon_index[data_index]
        if grid_row < 0 or grid_col < 0:
            continue
        else:
            out_vals[grid_row,grid_col].append(data_index)

    cdef float[:,:] rad_grid=np.empty_like(out_vals,dtype=np.float32)
    cdef float[:,:] lat_grid=np.empty_like(out_vals,dtype=np.float32)
    cdef float[:,:] lon_grid=np.empty_like(out_vals,dtype=np.float32)
    cdef int[:,:] count_grid=np.empty_like(out_vals,dtype=np.int32)
    rows,cols=out_vals.shape
    flat_chan31=rads.ravel()

    for row in range(rows):
        for col in range(cols):
            rad_list=out_vals[row,col]
            if len(rad_list)==0:
                rad_grid[row,col]=np.nan
                lat_grid[row,col]=np.nan
                lon_grid[row,col]=np.nan
            else:
                rad_vals=np.take(flat_chan31,rad_list)
                lat_vals=np.take(lats,rad_list)
                lon_vals=np.take(lons,rad_list)
                rad_grid[row,col]=np.mean(rad_vals)
                lat_grid[row,col]=np.mean(lat_vals)                    
                lon_grid[row,col]=np.mean(lon_vals)
                count_grid[row,col]=len(lon_vals)

    return np.asarray(lat_grid),np.asarray(lon_grid),np.asarray(rad_grid),np.asarray(count_grid)

def hist_2d(dict bin_xvals,dict bin_yvals):
    """
       calculate the 2-d histogram given two dictionaries produced by
       calls to fastbin.do_bins  

       parameters
       ----------

        bin_xvals: dictionary with bin_index vector from do_bins containing
                   the bin index for each datapoint and the number of
                   bins numbins

        bin_yvals: as for bin_xvals, but for the yvalues for the histogram

       returns:
       --------

         dictionary with the following keys:

            index_grid: numpy object array with shape [numbinsx,numbinsy] containing the
                     lists of datapoints that were binned into each bin

            count_grid: numpy float32 array with shape [numbinsx,numbinsy] containing the
                     counts in each bin

       example
       -------

             #bin a temperature vector into 20 1K  bins
             #starting at 270.
             bin_dict=obj.do_bins(temp_vec,20,270.,300.)
    """   
    out_vals=np.empty([bin_yvals['numbins'],bin_xvals['numbins']],dtype=np.object)
    x_index=bin_xvals['bin_index']
    y_index=bin_yvals['bin_index']
    xvals=bin_xvals['data_vec']
    yvals=bin_yvals['data_vec']
    cdef int numybins=bin_yvals['numbins']
    cdef int numxbins=bin_xvals['numbins']
    cdef int num_datapts=<int> y_index.size
    cdef int row,col, data_index, grid_row,grid_col,rows,cols
    cdef int[:] y_view=y_index
    cdef int[:] x_view=x_index
    for row in range(numybins):
        for col in range(numxbins):
            out_vals[row,col]=list()
    for data_index in range(num_datapts):
        grid_row=y_index[data_index]
        grid_col=x_index[data_index]
        if grid_row < 0 or grid_col < 0:
            continue
        else:
            out_vals[grid_row,grid_col].append(data_index)
    #
    # use float32 for counts so we can use np.nan
    # for plotting
    #
    cdef float[:,:] count_grid=np.zeros_like(out_vals,dtype=np.float32)
    rows,cols=out_vals.shape
    for row in range(rows):
        for col in range(cols):
            data_list=out_vals[row,col]
            if len(data_list)==0:
                count_grid[row,col]=0
            else:
                count_grid[row,col]=len(data_list)
    out_dict=dict(index_grid=out_vals,count_grid=np.asarray(count_grid))
    return out_dict
