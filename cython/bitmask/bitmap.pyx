#cython: embedsignature=True
import numpy as np
cimport numpy as np
from libc.stdint cimport int8_t

#try from libc.stdint cimport int64_t 

cdef extern from "bitmask.h":
   void readcloud_cpp(signed char* byteone,np.float32_t* maskout,int nvals)
   void readland_cpp(signed char* byteone,np.float32_t* landout,int nvals)
   void readglint_cpp(signed char* byteone,np.float32_t* glintout,int nvals)
   void readthin_cirrus_cpp(signed char* byteone,signed char* cirrusout,int nvals)
   void readhigh_cloud_cpp(signed char* byteone,signed char* highout,int nvals)


def getmask_zero(object bytezero):
    """
       http://modis-atmos.gsfc.nasa.gov/MOD35_L2/format.html
       http://modis-atmos.gsfc.nasa.gov/MOD35_L2/index.html
       http://modis-atmos.gsfc.nasa.gov/_specs/MOD35_L2.CDL.fs
       input is 2 dimensional numpy array of type np.int8
       containing 0 byte of the cloud mask

       output is a tuple of three 2 dimensional np.float32 arrays

       maskout has each pixel's cloud probability:

        0 = Cloud            
        1 = 66% prob. Clear  
        2 = 95% prob. Clear  
        3 = 99% prob. Clear

       landout has land/water values

       0=Water   
       1=Coastal 
       2=Desert  
       3=Land    

       glintout has the sun glint flag

       0=glint
       1=no glint
       
    """
    bytezero=np.ascontiguousarray(bytezero)
    cdef int nvals= bytezero.size
    #
    # create memoryview wrappers arround the input and output
    # numpy arrays and get c pointers to the start of the data
    # to pass to the c++ functions
    #
    cdef np.int8_t* dataPtr= <np.int8_t*> np.PyArray_DATA(bytezero)
    maskout=np.empty(bytezero.shape,dtype=np.float32)
    landout=np.empty(bytezero.shape,dtype=np.float32)
    glintout=np.empty(bytezero.shape,dtype=np.float32)
    cdef np.float32_t* maskPtr = <np.float32_t*> np.PyArray_DATA(maskout)
    cdef np.float32_t* landPtr = <np.float32_t*> np.PyArray_DATA(landout)
    cdef np.float32_t* glintPtr = <np.float32_t*> np.PyArray_DATA(glintout)
    #
    # call the c++ functions to read the cloud and 
    # land masks
    #
    readcloud_cpp(dataPtr, maskPtr,nvals)
    readglint_cpp(dataPtr, glintPtr,nvals)
    readland_cpp(dataPtr, landPtr,nvals)
    #
    # cast the memoryview objects back to numpy arrays
    # to return to python
    #
    out=(maskout,landout,glintout)
    return out


def getmask_one(object byteone):
    """
       input is 2 dimensional numpy array of type np.int8
       containing  byte 1 of the cloud mask

       output is a tuple of two 2 dimensional np.int8 arrays

       thinout has 0 if thin cirrus, 1 if not, so clear=True


       highout has 0 if high cloud, 1 if not, so clear=True

    """
    byteone=np.ascontiguousarray(byteone)
    cdef int nvals= byteone.size
    saveShape=byteone.shape
    cdef np.int8_t[:,::1] c_byte=byteone
    cdef np.int8_t* dataPtr=<int8_t*> &c_byte[0,0]
    cdef np.int8_t[:,::1] thinout=np.empty_like(byteone)
    cdef np.int8_t* thinPtr= &thinout[0,0]
    cdef np.int8_t[:,::1] highout=np.empty_like(byteone)
    cdef np.int8_t* highPtr= &highout[0,0]
    readthin_cirrus_cpp(dataPtr, thinPtr,nvals)
    readhigh_cloud_cpp(dataPtr, highPtr,nvals)
    out=(np.asarray(thinout),np.asarray(highout))
    return out




