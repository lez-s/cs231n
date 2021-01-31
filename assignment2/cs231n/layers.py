from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #print(x.shape)
    #x.shape=(2, 4, 5, 6)
    x_shape=x.reshape(x.shape[0],-1)
    out=np.dot(x_shape,w)+b
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #注意x是个k维数组

    x_shape=x.reshape(x.shape[0],-1)
    dw=np.dot(x_shape.T,dout)
    dx=np.dot(dout,w.T).reshape(x.shape)  #计算完成后把x缩放回去
    db=np.sum(dout,axis=0)  #注意db，之前没有计算过。计算所有行变成1行。

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #cache = x.copy()
    out=x.copy() #out=x不行
    out[out<=0]=0 #如果用out=x的话，这段代码会让x的值也改变

    #out = np.maximum(0,x)   #网上搜索的代码
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    #print(np.sum(cache))
    #print(np.sum(x))
    #input()
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx = dout.copy()
    #k=np.where(x<=0)

    dx[x <= 0] = 0
    
    #dx = (x>0) * dout #搜索的代码，功能和自己写的没有区别，出现问题的原因是forward的部分

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    running average是啥？？

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable                              #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        cache={}
        #num_batch=np.random.choice(N,size=min(N,100),replace = True) 
        #明明说了minibatch，为啥不用了呢？？？
        #x_batch=x[num_batch,:]
        #用了minibatch会导致x的行不一样，因为是放回地抽取。
        
        mean=np.mean(x, axis=0) #对每行求平均，变成一行。
        #为什么是对行求平均？因为是对所有sample进行归一化
        var=np.mean(np.square(x-mean),axis=0)
        running_mean=momentum*running_mean+(1-momentum)*mean
        running_var=momentum*running_var+(1-momentum)*var
        
        xi=(x-mean)/np.sqrt(var+eps)
        out=gamma*xi+beta
        bn_param["running_mean"] = running_mean
        bn_param["running_var"] = running_var
        cache['gamma'],cache['beta']=gamma,beta
        cache['mean'],cache['var'],cache['eps']=mean,var,eps
        cache['x'],cache['xi']=x,xi
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #running_mean/var是在test的时候用的？？ 因为测试时使用的mean/std不同，
        #测试时使用固定的mean/var，所以要用上面要计算running var/mean
        xi=(x-running_mean)/np.sqrt(running_var+eps)
        out=gamma*xi+beta
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    由于BN层是夹在中间的，所以要用反向传播计算梯度

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    gamma,beta=cache['gamma'],cache['beta']
    mean,var,eps=cache['mean'],cache['var'],cache['eps']
    x,xi=cache['x'],cache['xi']
    dxi=dout*gamma #此处没问题
    dvar=np.sum(dxi*(x-mean),axis=0)*-0.5*(var+eps)**(-1.5)
    dmean=np.sum(-dxi/np.sqrt(var+eps),axis=0)+dvar*np.mean((-2)*(x-mean),axis=0)
    dx=dxi/(np.sqrt(var+eps)) + 2*dvar*(x-mean)/x.shape[0] + dmean/x.shape[0] #这个地方m=x的行数
    dgamma=np.sum(dout*xi,axis=0)
    dbeta=np.sum(dout,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma,beta=cache['gamma'],cache['beta']
    mean,var,eps=cache['mean'],cache['var'],cache['eps']
    x,xi=cache['x'],cache['xi']
    dxi=dout*gamma #以下计算按照课件上的公式
    dvar=np.sum(dxi*(x-mean),axis=0)*-0.5*(var+eps)**(-1.5)
    dmean=np.sum(-dxi/np.sqrt(var+eps),axis=0)+dvar*np.mean((-2)*(x-mean),axis=0)
    dx=dxi/(np.sqrt(var+eps)) + 2*dvar*(x-mean)/x.shape[0] + dmean/x.shape[0] #这个地方m=x的行数
    dgamma=np.sum(dout*xi,axis=0)
    dbeta=np.sum(dout,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache={}
    mean=np.mean(x,axis=1).reshape(-1,1)
    var=np.mean(np.square(x-mean),axis=1).reshape(-1,1)
    xi=(x-mean)/np.sqrt(var+eps)
    out=gamma*xi+beta
    #感觉BN是把所有sample的值加成一行，LB是把所有dim的值加成一列

    #纯粹凑的，根本不知道为啥这么算。
    #mean=np.mean(x.T, axis=0) 
    #var=np.mean(np.square(x.T-mean),axis=0)
    #xi=(x.T-mean)/np.sqrt(var+eps)
    #out=gamma*xi.T+beta
    cache['gamma'],cache['beta']=gamma,beta
    cache['mean'],cache['var'],cache['eps']=mean,var,eps
    cache['x'],cache['xi']=x,xi
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma,beta=cache['gamma'],cache['beta']
    mean,var,eps=cache['mean'],cache['var'],cache['eps']
    x,xi=cache['x'],cache['xi']
    dxi=dout*gamma 
    dvar=np.sum(dxi*(x-mean),axis=1).reshape(-1,1)*-0.5*(var+eps)**(-1.5)
    dmean=np.sum(-dxi/np.sqrt(var+eps),axis=1).reshape(-1,1)+dvar*np.mean((-2)*(x-mean),axis=1).reshape(-1,1)
    dx=dxi/(np.sqrt(var+eps)) + 2*dvar*(x-mean)/x.shape[1] + dmean/x.shape[1] #此处m=x的列数
    dgamma=np.sum(dout*xi,axis=0)
    dbeta=np.sum(dout,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask=(np.random.rand(*x.shape)<p)/p  #*为解包数组中的元素
        #np.random.rand(*x.shape)>p 得到的是true false，默认true为1，false为0
        #print(x_drop)
        out = x*mask
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out=x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx=dout*mask  #就是把dropout掉的设为0

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #np.pd为填充数组
    pad, stride=conv_param['pad'],conv_param['stride']
    #H'=1+()
    F,HH,WW=w.shape[0],w.shape[2],w.shape[3]
    x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
    #第一个pad是对行操作（第一行和最后一行加pad），第二个是列。
    N,H,W=x.shape[0],x.shape[2],x.shape[3]
    H_dash=(H-HH)//stride+1
    W_dash=(W-WW)//stride+1
    v=np.zeros((N,F,H_dash,W_dash))
    #v[0,0,0,2]=9
    #print(H_dash)
    #input()
    for n in range(N):
      for f in range(F):
        for hh in range(H_dash):
          for ww in range(W_dash):
            s_w=ww*stride
            s_h=hh*stride
            #print(s_w,s_w+WW)
            v[n,f,hh,ww]=np.sum(x[n,:,s_h:s_h+HH,s_w:s_w+WW]*w[f,:,:,:])+b[f]
            #print('%s, %s, %s:%s, %s:%s'%(n,f,s_h,s_h+HH,s_w,s_w+WW))
            #注意，这里x的第二项是:,草稿的时候写对的，但是写程序的时候就错了。过程为书本上的
    
    out=v
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param=cache 
    pad, stride=conv_param['pad'],conv_param['stride']
    F,HH,WW=w.shape[0],w.shape[2],w.shape[3]
    dx,dw,db=np.zeros(x.shape),np.zeros(w.shape),np.zeros(b.shape)
    N,H,W=x.shape[0],x.shape[2],x.shape[3]
    H_dash=(H-HH)//stride+1
    W_dash=(W-WW)//stride+1
    
    for n in range(N):
      for f in range(F):
        for hh in range(H_dash):
          for ww in range(W_dash):
            s_w=ww*stride
            s_h=hh*stride
            dw[f,:,:,:] += x[n,:,s_h:s_h+HH,s_w:s_w+WW]*dout[n,f,hh,ww]
            #这里dout是一个数值,x是(C,HH,WW) dw(f)和dw[f,:,:,:]效果一样？？
            #dW是自己没写出来的。对forward里的求导理解不到位，因为W在移动，所以是所有的x*dout
            db[f]+=dout[n,f,hh,ww]  #dv对db求导为1*dout
            dx[n,:,s_h:s_h+HH,s_w:s_w+WW] += w[f,:,:,:]*dout[n,f,hh,ww] 
            #这里的dx也是参考来的。
            
    dx=dx[:,:,pad:H-pad,pad:W-pad]        
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape
    pool_height,pool_width,stride=pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    H_dash= 1 + (H - pool_height) // stride
    W_dash= 1 + (W - pool_width) // stride
    v=np.zeros((N,C,H_dash,W_dash))
    max_index={}

    for n in range(N):
      for h in range(H_dash):
        for w in range(W_dash):
          for c in range(C):
            s_w=w*stride
            s_h=h*stride
            mid_val=x[n,c,s_h:s_h+pool_height,s_w:s_w+pool_width]
            #midval为切出来的一个(pool_hight,pool_width)的矩阵
            v[n,c,h,w]=np.max(mid_val)
            max_index[n,c,h,w]=np.where(mid_val==v[n,c,h,w])
    out=v
    pool_param['max_index']=max_index
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=cache[0].shape
    x,pool_param=cache
    pool_height,pool_width,stride=pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    max_index=pool_param['max_index']
    H_dash= 1 + (H - pool_height) // stride
    W_dash= 1 + (W - pool_width) // stride
    dx=np.zeros((N,C,H,W))

    for n in range(N):
      for h in range(H_dash):
        for w in range(W_dash):
          for c in range(C):
            s_w=w*stride
            s_h=h*stride
            index=max_index[n,c,h,w]
            #dx[n,c,s_h:s_h+pool_height,s_w:s_w+pool_width]=(x[n,c,s_h:s_h+pool_height,s_w:s_w+pool_width]==val)*dout[n,c,h,w]
            #此代码为参考网络上的
            dx[n,c,s_h:s_h+pool_height,s_w:s_w+pool_width][index]=dout[n,c,h,w]
            #自己原来写的是val*dout，这里不需要乘val，原函数不为零处求导的结果为1.
            #这里要注意下，原函数求导值老理解错误

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape
    #网上发现一个更好的算法
    x_new=x.transpose(0,3,2,1).reshape(-1,C)  #把x(N,C,H,W)变成(N,W,H,C)
    out,cache=batchnorm_forward(x_new,gamma,beta,bn_param)
    out=out.reshape(N,W,H,C).transpose(0,3,2,1)
    '''
    out=np.zeros(x.shape)
    cache={}
    for w in range(W):
      for h in range(H):
        #print(x[:,:,h,w].shape,gamma.shape)
        #input()
        out[:,:,h,w], cache[h,w]=batchnorm_forward(x[:,:,h,w], gamma, beta, bn_param)
        #主要由于把gamma的shape设为c了，所有输入x的第二个纬度必须要是c
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #参考网上代码实现
    N,C,H,W=dout.shape
    dout_new=dout.transpose(0,3,2,1).reshape(-1,C)  ##把(N,C,H,W)变成(N,W,H,C)
    dx, dgamma, dbeta=batchnorm_backward_alt(dout_new,cache)
    dx=dx.reshape(N,W,H,C).transpose(0,3,2,1)

    '''自己写的dx计算出来应该有错误，dgamma和dbeta对
    N,C,H,W=dout.shape
    dx,dgamma,dbeta=np.zeros(dout.shape),np.zeros(C),np.zeros(C)
    for w in range(W):
      for h in range(H):
        dx[:,:,h,w], dgamma1, dbeta1 = batchnorm_backward_alt(dout[:,:,h,w], cache[h,w])
        #print(h,w,dx[:,:,h,w].shape)
        #input()
        #dx[:,:,h,w]+=dx1
        dgamma+=dgamma1
        dbeta+=dbeta1
        #print(dgamma)
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache={}
    N,C,H,W=x.shape
    C_sub=C//G
    cache['G'],cache['gamma'],cache['beta']=G,gamma,beta
    out=np.zeros(x.shape)
    #方法论就是和上面的一样，把4维的压缩到2维后计算，然后缩放回4维
    for i in range(G):
      x_new=x[:,i*C_sub:(i+1)*C_sub,:,:].transpose(0,3,2,1).reshape(-1,C_sub)  #(NCHW)→ (NWHC)
      gamma_sub=gamma[:,i*C_sub:(i+1)*C_sub,:,:].transpose(0,3,2,1).reshape(-1,C_sub) 
      beta_sub=beta[:,i*C_sub:(i+1)*C_sub,:,:].transpose(0,3,2,1).reshape(-1,C_sub) 
      #把x，gamma，beta降维
      out_2d,cache[i]=layernorm_forward(x_new, gamma_sub, beta_sub, gn_param)
      out[:,i*C_sub:(i+1)*C_sub,:,:]=out_2d.reshape(N,W,H,C_sub).transpose(0,3,2,1)
    
    ''' 网上的做法，和自己的不一样，可以看看是怎么实现的。
    N,C,H,W = x.shape
    x_group = np.reshape(x,(N,G,C//G,H,W)) #分为C//G组
    mean = np.mean(x_group,axis = (2,3,4),keepdims = True) #求均值
    var = np.var(x_group,axis = (2,3,4),keepdims = True)  #求方差
    x_groupnorm = (x_group - mean) / np.sqrt(var + eps)  #归一化
    x_norm = np.reshape(x_groupnorm,(N,C,H,W))  #还原维度
    out = x_norm * gamma + beta  #还原C
    cache = (G,x,x_norm,mean,var,beta,gamma,eps)
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    G,gamma,beta=cache['G'],cache['gamma'],cache['beta']
    N,C,H,W=dout.shape
    N_beta,C_beta,H_beta,W_beta=beta.shape  #注意此处的beta和gamma的维度和x/dout不同
    C_sub=C//G
    dx,dgamma,dbeta=np.zeros(dout.shape),np.zeros(gamma.shape),np.zeros(beta.shape)
    for i in range(G):
      dout_sub=dout[:,i*C_sub:(i+1)*C_sub,:,:].transpose(0,3,2,1).reshape(-1,C_sub)
      dx_2d,dgamma_2d,dbeta_2d=layernorm_backward(dout_sub, cache[i])
      dx[:,i*C_sub:(i+1)*C_sub,:,:]=dx_2d.reshape(N,W,H,C_sub).transpose(0,3,2,1)
      dgamma[:,i*C_sub:(i+1)*C_sub,:,:]=dgamma_2d.reshape(N_beta,W_beta,H_beta,C_sub).transpose(0,3,2,1)
      dbeta[:,i*C_sub:(i+1)*C_sub,:,:]=dbeta_2d.reshape(N_beta,W_beta,H_beta,C_sub).transpose(0,3,2,1)
    
    '''网上的代码，有空可以看看实现方法
    N,C,H,W = dout.shape
    G,x,x_norm,mean,var,beta,gamma,eps = cache
    dbeta = np.sum(dout,axis = (0,2,3),keepdims = True)
    dgamma = np.sum(dout * x_norm,axis = (0,2,3),keepdims = True)
    
    #计算dx_group
    #dx_groupnorm
    dx_norm = dout * gamma
    dx_groupnorm = dx_norm.reshape((N,G,C//G,H,W))
    #dvar
    x_group = x.reshape((N,G,C//G,H,W))
    dvar = np.sum(dx_groupnorm * -1.0 / 2 * (x_group - mean) / (var + eps) ** (3.0/2),axis = (2,3,4),keepdims = True)
    #dmean
    n_group = C // G * H * W
    dmean1 = np.sum(dx_groupnorm * -1.0 / np.sqrt(var + eps),axis = (2,3,4),keepdims = True)
    dmean2_var = dvar * -2.0 / n_group * np.sum(x_group - mean,axis = (2,3,4),keepdims = True)
    dmean = dmean1 + dmean2_var
    #dx_group
    dx_group1 = dx_groupnorm * 1.0 / np.sqrt(var + eps)
    dx_group2_mean = dmean * 1.0 / n_group
    dx_group3_var = dvar * 2.0 / n_group * (x_group - mean)
    dx_group = dx_group1 + dx_group2_mean + dx_group3_var
    
    #还原dx
    dx = dx_group.reshape(N,C,H,W)
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
