from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) #(3073, 10)
    num_train=X.shape[0]
    num_class=W.shape[1]


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
      scores=np.dot(X[i],W)
      scores -= np.max(scores) #进行平移防止数值过大
      scores_pob=np.exp(scores)/np.sum(np.exp(scores)) #计算softmax
      for j in range(num_class):
        
        if j==y[i]:
          loss+= -np.log(scores_pob[j]) #这里的log其实是ln，log10才是log
          dW[:,j] += (scores_pob[j]-1)*X[i] #反正就是求导结果，怎么算的还不知道。
        else:
          dW[:,j] += scores_pob[j]*X[i]
          
    loss /= num_train #不要忘记平均
    loss += reg*np.sum(W*W) #不要忘记正则化
    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_class=W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=np.dot(X,W)
    scores -= np.max(scores,axis=1).reshape(-1,1) #平移
    sum_e=np.sum(np.exp(scores),axis=1).reshape(-1,1) 
    #刚开始算的时候把axis搞错了。axis=1位所有列相加。以及忘记reshape了

    scores_pob=np.exp(scores)/sum_e
    scores_pob_right=scores_pob[np.arange(num_train),y] #把所有正确的分类的score_pos相加
    loss=np.sum(-np.log(scores_pob_right))/num_train #用log计算loss
    loss += reg*np.sum(W*W)
    scores_pob_dW = scores_pob #新建一个score_possible矩阵
    scores_pob_dW[np.arange(num_train),y] -= 1 #把所有正确的列的概率-1
    
    dW= np.dot(X.T,scores_pob_dW)/num_train + 2*reg*W
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
