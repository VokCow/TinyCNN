import numpy as np

def sigm(x,bck=False):
    if bck: return x*(1-x)
    return 1/(1-np.e**(-x))

''' 
inputs (xs): flat mapped versions of forward_slash (i.e. /) and backwards_slash (i.e \). 
outputs (ys): we'll identify forward slash as the vector (1 0) adn backwards slash as (0 1)
'''

train_imgs=[[1,-1,-1,1],[-1,1,1,-1]]
train_labels=[[1,0],[0,1]]

test_imgs=[[1,1,-1,1],[-1,-1,1,-1],[1,-1,-1,-1],[1,-1,-1,1],[-1,-1,-1,1],[-1,1,1,-1]]
test_labels=[[1,0],[0,1],[1,0],[1,0],[1,0],[0,1]]

'''
now let's fill a matrix of random weights. Dimension of this matrix must be 2x4 so when we apply it 
to our 4x1 flat mapped image we obtain the feature 2x1 vector.
'''
np.random.seed(1)

W=np.random.random((2,4))*2-1

forw_img=np.array([train_imgs[0]]).T
bck_img=np.array([train_imgs[1]]).T

forw_label=np.array([train_labels[0]]).T
bck_label=np.array([train_labels[1]]).T

for i in range(5000):
    out=sigm(np.dot(W,forw_img),False)
    err=out-forw_label
    W+=np.dot(err*sigm(out,True),forw_img.T)
    out=sigm(np.dot(W,bck_img),False)
    err=out-bck_label
    W+=np.dot(err*sigm(out,True),bck_img.T)
    print(i)
