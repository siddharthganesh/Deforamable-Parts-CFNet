
import numpy as np


@profile
def dt1d(vals,out_vals,I,step,shift,n,a,b):
    for i in range(0,n):
        max_val=-float('inf')
        argmax=0
        first=max(0,i-shift)
        last=min(n-1,i+shift)
        for j in range(first,last+1):
            val = vals[j*step] - a*(i-j)*(i-j) - b*(i-j)
            if val>max_val:
                max_val=val
                argmax=j
        out_vals[i*step]=max_val
        I[i*step]=argmax
    return I,out_vals

@profile
def dt2d(scoreMap,w,shift):
    cdef float ax = w[0]
    cdef float bx = w[1]
    cdef float ay = w[2]
    cdef float b_y = w[3]
    [n1,n2]=np.shape(scoreMap)
    tmpOut = np.zeros((n1, n2))
    tmpIy = np.zeros((n1, n2))
    Ix = np.zeros((n1, n2))
    Iy = np.zeros((n1, n2))
    for x in range(0,n2):
        tmpIy[:,x],tmpOut[:,x] = dt1d(scoreMap[:,x],tmpOut[:,x],tmpIy[:,x],1, shift, n1, ay, b_y)
    for y in range(0,n1):
        Ix[y,:],scoreMap[y,:] = dt1d(tmpOut[y,:],scoreMap[y,:],Ix[y,:],1, shift, n2, ax, bx)
    
    
    for x in range(0,n2): 
        for y in range(0,n1):
            t = Ix[y,x]
            Iy[y, x] = tmpIy[y, int(Ix[y,x])]
    return scoreMap
