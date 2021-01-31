from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(    #初始化一堆参数
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        #randn为返回正态分布。
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2']= weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b2']=np.zeros(num_classes) #b是按行加的？

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out_mid1, cache_mid1= affine_relu_forward(X, W1, b1)
        scores, cache_mid2=affine_forward(out_mid1,W2,b2)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx=softmax_loss(scores,y) #loss要算平均值。但是softmax函数里已经算了平均值
        loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)) #！注意要×0.5，上面说了
        dx_2, dW2, db2 = affine_backward(dx, cache_mid2)
        _, dW1, db1 = affine_relu_backward(dx_2, cache_mid1) 
        #注意此处为dx2，而不是dw2，dw2保存到grads里去了，不参与下次运算
        dW2 += self.reg*W2
        dW1 += self.reg*W1
        grads['W1'],grads['W2'] = dW1, dW2
        grads['b1'],grads['b2'] = db1, db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.input_dim=input_dim
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        num_layers=self.num_layers

        self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])
        for l in range(1,num_layers-1):  #-1的原因是隐藏层为n-1，起点为0。
          self.params['W%s' % (l+1)] = weight_scale*np.random.randn(hidden_dims[l-1], hidden_dims[l])
          self.params['b%s' % (l+1)] = np.zeros(hidden_dims[l])

        self.params['W%s' % num_layers] = weight_scale*np.random.randn(hidden_dims[num_layers-2],num_classes)
        #最后一个affine不是隐藏层，如果一共有n层，隐藏层就是n-1层。因为起点是0，所以是n-2
        self.params['b%s' % num_layers] = np.zeros(num_classes)
        
        if self.normalization == 'batchnorm' or self.normalization=='layernorm': 
          #如果为batchnorm就初始化gamma参数
          for l in range(num_layers-1):
            self.params['gamma%s'%(l+1)]=np.random.randn(hidden_dims[l])
            self.params['beta%s'%(l+1)]=np.random.randn(hidden_dims[l])
        #print(self.params.keys())
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout: 
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]
        
        self.ln_param={}
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)



    def loss(self, X, y=None):
        #以下为自己添加的batchnorm
        def affine_batchnorm_relu_forward(x,w,b,gamma,beta,bn_param):
            out_affine, fc_cache = affine_forward(x,w,b)
            out_bn,cache_bn = batchnorm_forward(out_affine,gamma,beta,bn_param)
            out, cache_relu=relu_forward(out_bn)
            cache=(fc_cache,cache_bn,cache_relu)
            #print('forward',len(cache))
            return out, cache

        def affine_batchnorm_relu_backward(dout,cache):
            fc_cache,cache_bn,cache_relu = cache
            dout_relu = relu_backward(dout,cache_relu)
            dout_batchnorm,dgamma,dbeta = batchnorm_backward_alt(dout_relu,cache_bn)
            dx,dw,db = affine_backward(dout_batchnorm,fc_cache)
            return dx,dw,db,dgamma,dbeta

        def affine_ln_relu_forward(x,w,b,gamma,beta,ln_param):
            out_affine, fc_cache = affine_forward(x,w,b)
            out_ln,cache_ln = layernorm_forward(out_affine,gamma,beta,ln_param)
            out, cache_relu=relu_forward(out_ln)
            cache=(fc_cache,cache_ln,cache_relu)
            return out, cache

        def affine_ln_relu_backward(dout,cache):
            fc_cache,cache_ln,cache_relu = cache
            dout_relu = relu_backward(dout,cache_relu)
            dout_ln,dgamma,dbeta = layernorm_backward(dout_relu,cache_ln)
            dx,dw,db = affine_backward(dout_ln,fc_cache)
            return dx,dw,db,dgamma,dbeta


        #以上结束
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:  
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        num_layers=self.num_layers
        cache_mid={}  #建立个字典保存中间值
        cache_dropout={}
        out=X #这么做好像更简单
        #if self.normalization == "normalization":
        #  out,cache_mid[1]=affine_relu_forward(X,self.params['W1'],self.params['b1'])
        #elif self.normalization =='batchnorm':
        #  out,cache_mid[1]=affine_batchnorm_relu_forward(X,self.params['W1'],self.params['b1'],{'mode':mode})
        #由于第一层的输入值为x，所以要单独拿出来
        
        if self.normalization == 'batchnorm':
          for i in range(num_layers-1):
            out,cache_mid[i+1]=affine_batchnorm_relu_forward(out,self.params['W%s'%(i+1)],self.params['b%s'%(i+1)],self.params['gamma%s'%(i+1)],self.params['beta%s'%(i+1)],self.bn_params[i])
            if self.use_dropout:  #判断是否添加dropout。比起在各层插入判断，感觉有更优雅的实现方式
              out,cache_dropout[i+1]=dropout_forward(out,self.dropout_param)
        elif self.normalization == None:
          for i in range(num_layers-1):
            out,cache_mid[i+1]=affine_relu_forward(out,self.params['W%s'%(i+1)],self.params['b%s'%(i+1)])
            if self.use_dropout:  #判断是否添加dropout。比起在各层插入判断，感觉有更优雅的实现方式
              out,cache_dropout[i+1]=dropout_forward(out,self.dropout_param)
            #把上一层的out，和保存在params里的参数拿出来计算前向传播
        elif self.normalization=='layernorm':
          for i in range(num_layers-1):
            out,cache_mid[i+1]=affine_ln_relu_forward(out,self.params['W%s'%(i+1)],self.params['b%s'%(i+1)],self.params['gamma%s'%(i+1)],self.params['beta%s'%(i+1)],self.ln_param)
            if self.use_dropout:  #判断是否添加dropout。比起在各层插入判断，感觉有更优雅的实现方式
              out,cache_dropout[i+1]=dropout_forward(out,self.dropout_param)
        
        scores,cache_mid[num_layers]=affine_forward(out,self.params['W%s'%num_layers],self.params['b%s'%num_layers])
        #最后一层由于只有affine，所以单独拿出来
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        grads_x={} #由于grads里不能包括x，所以把x的值拿出来
        loss, dx=softmax_loss(scores,y)
        for i in range(num_layers):
          loss+= 0.5*self.reg*(np.sum(self.params['W%s'%(i+1)]*self.params['W%s'%(i+1)]))
          #计算正则化。所有W*W的值，然后相加
        
        grads_x['x%s'%num_layers],grads['W%s'%num_layers],grads['b%s'%num_layers]=affine_backward(dx,cache_mid[num_layers])
        grads['W%s'%num_layers] += self.reg*self.params['W%s'%num_layers]
        #由于第一层的输入是dx，所以单独拿出来。感觉用if语句判断也行？？
        if self.normalization==None:
          for i in range(num_layers-1,0,-1): #倒序循环
            if self.use_dropout:  #判断是否添加dropout。比起在各层插入判断，感觉有更优雅的实现方式
              grads_x['x%s'%(i+1)]=dropout_backward(grads_x['x%s'%(i+1)],cache_dropout[i])
            grads_x['x%s'%i],grads['W%s'%i],grads['b%s'%i]=affine_relu_backward(grads_x['x%s'%(i+1)],cache_mid[i])
            #计算dx，dw，db的affinerelu反向传播值等于，输入为上一层的dx和本层cache中的值。
            grads['W%s'%i] += self.reg*self.params['W%s'%i]
        
        elif self.normalization=='batchnorm':
          for i in range(num_layers-1,0,-1): #倒序循环
            if self.use_dropout:  #判断是否添加dropout。比起在各层插入判断，感觉有更优雅的实现方式
              grads_x['x%s'%(i+1)]=dropout_backward(grads_x['x%s'%(i+1)],cache_dropout[i])
            grads_x['x%s'%i],grads['W%s'%i],grads['b%s'%i],grads['gamma%s'%i],grads['beta%s'%i]=affine_batchnorm_relu_backward(grads_x['x%s'%(i+1)],cache_mid[i])
            grads['W%s'%i] += self.reg*self.params['W%s'%i]
        
        elif self.normalization=='layernorm':
          for i in range(num_layers-1,0,-1): 
            if self.use_dropout:  #判断是否添加dropout。比起在各层插入判断，感觉有更优雅的实现方式
              grads_x['x%s'%(i+1)]=dropout_backward(grads_x['x%s'%(i+1)],cache_dropout[i])
            grads_x['x%s'%i],grads['W%s'%i],grads['b%s'%i],grads['gamma%s'%i],grads['beta%s'%i]=affine_ln_relu_backward(grads_x['x%s'%(i+1)],cache_mid[i])
            grads['W%s'%i] += self.reg*self.params['W%s'%i]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
