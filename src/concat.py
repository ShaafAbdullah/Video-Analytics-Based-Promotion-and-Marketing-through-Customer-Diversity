import numpy as np

a=np.array([0,0,0,0,0,0,0,0])
b=np.array([0,0,0,0,0,0,0,0,1])
array3=np.concatenate((a,b),axis=0)

print (array3)
