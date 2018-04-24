#######################################################################################
# Neural network framework to play around with
# Author: Manuel Hass
# 2018
# 
#######################################################################################
try:
    import numpy as np
    numpy = np
except ImportError:
    print ('ERROR -> MODULE MISSING: numpy ')


################################# loss functions #######################################
def ce(y,yt,dev=False): ############ not robust !!
    '''
        cross entropy
        argmax stuff maybe otherwise near zero logs are silly
    '''
    if (dev==True):
        return (yt-y)
    loss = [(yt[i]).dot(np.log(y[i])) for i in range(y.shape[0])]
    loss = np.array(loss) * -1
    return np.sum(loss)   /(loss.shape[0]*1.)

def bce(ya,yta,dev=False):  ############ not robust !!
    '''
        binary cross entropy
    '''
    if (dev==True):
        ya = (ya)
        yta = (yta)
        return ((yta-ya)/((1-yta)*yta))
    return (-(np.sum(ya*np.log(yta)+(1.-yta)*np.log(1.-yta))/(yta.shape[0]*2.0)))

def qef(ya,yta,dev=False):
    '''
        quadratic error function ||prediction-target||²
    '''
    if (dev==True):
        return (yta-ya) 
    return np.sum((yta-ya)**2)/(yta.shape[0]*2.0)

def phl(y,yt,dev=False,delta=1.):
    '''
        subquadratic error function (pseudo huber loss)
    '''
    a = (yt-y)
    if (dev==True):
        return  a/( np.sqrt(a**2/delta**2 +1) ) 
    return np.sum((delta**2)*(np.sqrt(1+(a/delta)**2)-1)/(yt.shape[0]*2.0))


###################### regularization ####################################################
def L2_norm(lam,a):  
    '''
        2-Norm regularizer
    '''
    return lam*a

def L1_norm(lam,a):
    '''
        1-Norm regularizer
    '''
    return lam*np.sign(a)


###################### activation  ####################################################
def f_elu(a,dev=False):
    '''
        exponential linear unit
            ~softplus [0,a]
    '''
    if dev:
        return np.where(a>=0.,f_elu(a)+a,1)
    return np.where(a>=0.,a*(np.exp(a)-1),a)

######################## ############# ############ ########## #
def f_softmax(a,dev=False):
    if (dev==True):
        x = f_softmax(a)
        return x*(1-x)
    a = np.nan_to_num(a)
    exp = np.nan_to_num(np.exp(a-np.max(a)))
    sum_exp = ((np.sum(exp,axis=0)))
    return (exp / sum_exp)
############### ################## ################ ########### #

'''
def f_softmax(a,dev=False):
    
        softmax transfer function 
            sigmoidal [0,1]
    

    if (dev==True):
        return f_softmax(a)*(1-f_softmax(a))
    return  np.exp(a)/ np.sum(np.exp(a))
'''

def f_lgtr(a,dev=False):
    '''
        (robust) logistic transfer function 
            sigmoidal [0,1]
    '''
    if (dev==True):
        return (1-np.tanh(a/2.)**2)/2.
    return  (np.tanh(a/2.)+1)/2.
 
def f_stoch(a,dev=False):
    '''
        stochastic transfer function 
            activates if activated input > ~Uniform
            binary [0,1]
    '''
    if (dev==True):
        return np.zeros(a.shape)  
    x = f_lgtr(a,dev=False)
    rand = np.random.random(x.shape)
    return  np.where(rand < x,1,0)

def f_tanh(a,dev=False):
    '''
        hyperbolic tangent transfer function 
            sigmoidal [-1,1]
    '''
    if (dev==True):
        return (1-np.tanh(a)**2)
    return  np.tanh(a)

def f_atan(a,dev=False):
    '''
        arcus tangent transfer function 
            sigmoidal [-pi/2, pi/2]
    '''
    if (dev==True):
        return (1/(a**2+1))
    return  np.arctan(a)

def f_sp(a,dev=False):
    '''
        softplus transfer function 
            [0,a]

            ### kinda clip it...to make more robust
    '''
    if (dev==True):
        return np.exp(a)/(np.exp(a)+1.)
    return  np.log(np.exp(a)+1.)
    
def f_relu(a,dev=False):
    '''
        rectified linear transfer function 
            [0,a]
    '''
    if (dev==True):
        return np.maximum(0,np.sign(a)) 
    return  np.maximum(0.0,a)

def f_leaky(a,dev=False,leak=0.01):
    '''
       leaky rectified linear transfer function 
            [-leak*a,a]
    '''
    if (dev==True):
        signs = np.sign(a)

        return np.where(signs>0.,signs,leak*signs) 
    return  np.where(a>0.,a,leak*a)
 
def f_bi(a,dev=False):
    '''
        bent identity transfer function
    '''
    if (dev==True):
         return a / ( 2.0*np.sqrt(a**2+1) ) + 1
    return  (np.sqrt(a**2+1)-1)/2.0 + a

def f_iden(a,dev=False):
    '''
        identity transfer function 
    '''
    if (dev==True):
         return np.ones(a.shape)
    return  a

def f_bin(a,dev=False):
    '''
        binary step transfer function 
    '''
    if (dev==True):
         return np.zeros(a.shape) 
    return  np.sign(f_relu(a))


############################# utils ######################################
### input / output processing
def one_hot(targets,smooth=False):
    '''
        input: discrete labels (number, string, etc.)
        output: binary numpy array (size = #unique classes)
    '''
    classes =  np.unique(targets.T)
    binarycoded = []
    for i in classes:
        binarycoded +=  [np.where(targets==i,1,0)[0]]
    out = np.array(binarycoded).T
    if smooth:
        # one side label smoothind
        out = out+.8 +.1
    else:
        return out

def hot_one(targets):
    '''
        input: binary array
        output: discrete labels (numbers)
    '''
    return np.argmax(np.array(targets).T,axis=0).reshape(-1,1)


############################ LAYER ########################################### 
class layer:
    '''
    actiavtion layer for model building:
        layer(input_dimension,number_of_nodes)

    parameters:
        f   : activation function
        w   : weights

        reg : regularizer function
        lam : regularizer lambda
        eta : learning rate
        
        opt : optimizer ('Adam','RMSprop','normal')
        eps : "don't devide by zero!!"
        b1  : momentumparameter for 'Adam' optimizer
        b2  : momentumparameter for 'RMSprop' and 'Adam' optimizer
        m1  : momentum for 'Adam' optimizer
        m2  : momentum for 'RMSprop' and 'Adam' optimizer

        count: number of updates 

    '''
    def __init__(self,in_dim,nodes=32,no_bias=False): 

        #activation and weights
        self.no_bias = no_bias
        self.f = f_relu
        # by default Xavier init
        #np.random.randn(nodes, in_dim) / np.sqrt(in_dim)#
        #np.random.randn(nodes, in_dim+1) / np.sqrt(in_dim+1)#
        if self.no_bias: self.w = np.random.uniform(-.1,.1,(nodes,in_dim))
        else: self.w = np.random.uniform(-.1,.1,(nodes,in_dim+1))
        ### Xavier init: 
        #w = np.random.randn(neurons, input_dimension) / np.sqrt(input_dimension)

        #momentum 
        self.m1 = np.random.uniform(0.1,1,self.w.shape)
        self.m2 = np.random.uniform(0.1,1,self.w.shape)
        self.b1 = 0.9   # Adam, if b1 = 0. -> Adam = RMSprop
        self.b2 = 0.99
        self.opt = 'Adam'
        self.eps = 1e-8

        #regularizer
        self.reg = L2_norm
        self.lam = 1e-7

        #learning
        self.count = 0
        self.eta = 5e-4

    def forward(self,input_):
        '''
            forward pass (computes activation)
                return: activation(input * weights[+ bias])
        '''
        ##### IF no_bias != True  :
        if self.no_bias: self.x1 = input_
        else: self.x1 = np.vstack((input_.T,np.ones(input_.shape[0]))).T
        #print('fw x1: ',self.x1.shape)
        self.h1 = np.dot(self.x1,self.w.T).T
        #print('fw h1: ',self.h1.shape)
        self.s = self.f(self.h1)
        #print('fw s: ',self.s.shape)
        return self.s.T

    def backward(self,L_error):
        '''
            backward pass (computes gradient)
                return: layer delta
        '''
        #print('L_error :  ',L_error.shape)
        self.L_grad = L_error* self.f(self.h1,True).T
        #print('L_grad :  ',self.L_grad.shape)
        self.delta_W = -1./(self.x1).shape[0] * np.dot(self.L_grad.T,self.x1) - self.reg(self.lam,self.w)
        if self.no_bias: 
            return np.dot(self.w.T,self.L_grad.T).T
        else: return np.dot(self.w.T[1:],self.L_grad.T).T

    def update(self):   
        '''
            update step (updates weights & momentum)
        '''   
        self.m1 = self.b1*self.m1 + (1-self.b1)*self.delta_W
        self.m2 = self.b2*self.m2 + (1-self.b2)*self.delta_W**2
        if(self.opt=='RMSprop'):
            self.w += self.eta* self.delta_W / (np.sqrt(self.m2) +self.eps)
        if (self.opt=='Adam'):
            self.w += self.eta* self.m1 / (np.sqrt(self.m2) +self.eps)
        if(self.opt=='normal'):
            self.w += self.eta* self.delta_W
        self.count += 1

    def reset(self):
        '''
            weights & momentum reset
        '''
        self.w = np.random.uniform(-.7,.7,(nodes,in_dim+1))
        self.m1 = np.random.uniform(0.,1,self.w.shape)
        self.m2 = np.random.uniform(0.,1,self.w.shape)

class conv_layer:
    def __init__(self,filterwidth=3,filterheight=3,filterchannel=1,nodes=5 ,stride=1, padding=1,no_bias=False,flat_out=False):
        
        
         #activation and weights
        
        self.flat_out = flat_out
        self.no_bias = no_bias
        self.f = f_tanh  ### not in use
        
        self.w = np.random.uniform(-1,1,(nodes,filterchannel,filterheight,filterwidth))
        self.b = np.random.uniform(-1,1,(nodes,1))
        self.stride = stride
        self.padding = padding
        
        #momentum 
        self.m1 = np.random.uniform(0.1,1,self.w.shape)
        self.m2 = np.random.uniform(0.1,1,self.w.shape)
        self.m1b =np.random.uniform(0.1,1,self.b.shape)
        self.m2b =np.random.uniform(0.1,1,self.b.shape)
        self.b1 = 0.9   # Adam, if b1 = 0. -> Adam = RMSprop
        self.b2 = 0.99
        self.opt = 'Adam'
        self.eps = 1e-8

        #regularizer
        self.reg = L2_norm
        self.lam = 1e-7

        #learning
        self.count = 0
        self.eta = 5e-4
        
    
        self.input_flat = None
        self.input_shape = None
    
    def forward(self,input_):
        
        n_filters, d_filter, h_filter, w_filter = self.w.shape
        self.input_shape = input_.shape
        n_x, d_x, h_x, w_x = self.input_shape
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride +1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride +1

        if (not h_out.is_integer() or not w_out.is_integer()):
            raise Exception('decimal pixel output dimension')

        h_out, w_out = int(h_out), int(w_out)
        self.input_flat = im2col_indices(input_, h_filter, w_filter, padding=self.padding, stride=self.stride)
        W_flat = self.w.reshape(n_filters, -1)
    
        out = np.dot(W_flat, self.input_flat) 
        if not self.no_bias: out += self.b
        out = out.reshape(n_filters,h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        self.outer_shape = out.shape
        if self.flat_out: out= out.reshape(out.shape[0],-1)
        #print(out.shape)
        return out


    def backward(self, L_error):
        
        
        n_filter, d_filter, h_filter, w_filter = self.w.shape

        if self.flat_out: L_error =  L_error.reshape(self.outer_shape)#L_error.reshape(L_error.shape[0],self.w.shape[0] ,self.input_shape[2],self.input_shape[3])

        self.db= np.sum(L_error, axis=(0, 2, 3))[:,True]
    
        dout_reshaped = L_error.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = np.dot(dout_reshaped, self.input_flat.T) 
        dW += self.reg(self.lam,self.w.reshape(dW.shape))
        dW = dW.reshape(self.w.shape)
        W_reshape = self.w.reshape(n_filter, -1) 
        grad_flat = np.dot(W_reshape.T , dout_reshaped)  
        grad = col2im_indices(grad_flat, self.input_shape, h_filter, w_filter, padding=self.padding, stride=self.stride)
        self.dW = dW 
        
        return grad
    
    def update(self):
        '''
            update step (updates weights & momentum)
        '''   
        self.m1b = self.b1*self.m1b + (1-self.b1)*self.db
        self.m2b = self.b2*self.m2b + (1-self.b2)*self.db**2
        self.m1 = self.b1*self.m1 + (1-self.b1)*self.dW
        self.m2 = self.b2*self.m2 + (1-self.b2)*self.dW**2
        if(self.opt=='RMSprop'):
            self.w -= self.eta* self.dW / (np.sqrt(self.m2) +self.eps)
            self.b -= self.eta* self.db / (np.sqrt(self.m2b) +self.eps)
        if (self.opt=='Adam'):
            self.w -= self.eta* self.m1 / (np.sqrt(self.m2) +self.eps)
            self.b -= self.eta* self.m1b / (np.sqrt(self.m2b) +self.eps)
        if(self.opt=='normal'):
            self.w -= self.eta* self.dW
            self.b -= self.eta* self.db
        self.count += 1

class batchnorm_layer:
    '''
    Batch norm layer 
    '''
    def __init__(self,in_dim,noise=False,conv=True):
        
        self.conv = conv
        if self.conv:  self.gamma = np.ones((1,in_dim,1,1))
        else: self.gamma = np.ones((1,in_dim))
        if self.conv: self.beta = np.zeros((1,in_dim,1,1))
        else:self.beta = np.zeros((1,in_dim))
        self.epsilon = 1e-5
        
        
        self.training = True
        self.rng_mean = 0.
        self.rng_std = 1.
        
        #momentum 
        self.m1 = np.random.uniform(0.1,1,self.gamma.shape)
        self.m2 = np.random.uniform(0.1,1,self.gamma.shape)
        self.m1b =np.random.uniform(0.1,1,self.beta.shape)
        self.m2b =np.random.uniform(0.1,1,self.beta.shape)
        self.b1 = 0.9   # Adam, if b1 = 0. -> Adam = RMSprop
        self.b2 = 0.99
        self.opt = 'Adam'
        self.eps = 1e-8

        #learning
        self.count = 0
        self.eta = 5e-4
        
    def forward(self, input_):
        
        x = input_
        if self.training:
            if self.conv: 
                self.x_m = (x - (1./(x.shape[0]*x.shape[2]*x.shape[3]))) * np.sum(x,axis=(0,2,3)).reshape(1,x.shape[1],1,1)
                self.rng_mean = (.9) * self.rng_mean + (.1) * np.mean(x)
                self.x_var = (1./(x.shape[0]*x.shape[2]*x.shape[3])) * np.sum((self.x_m)**2.,axis=(0, 2, 3)).reshape(1, x.shape[1], 1, 1)
                self.x_std = np.sqrt(self.x_var+self.epsilon)
                self.rng_std = (.9) * self.rng_std + (.1) * self.x_std
                x_hat = self.x_m / self.x_std
            else:
                self.x_m = (x - np.mean(x,axis=0))
                self.rng_mean = (.9) * self.rng_mean + (.1) * np.mean(x)                
                self.x_std = np.sqrt( (1./x.shape[0]) * np.sum(np.power(self.x_m,2),axis=0) + self.epsilon )
                self.rng_std = (.9) * self.rng_std + (.1) * self.x_std          
                x_hat = self.x_m / self.x_std
        else:
            x_hat = (x-self.rng_mean)/self.rng_std
        

        if self.conv: 
            out = self.gamma.reshape(1,x.shape[1],1,1) * x_hat + self.beta.reshape(1,x.shape[1],1,1)
        else: 
            out = self.gamma * x_hat + self.beta

        return out
    
    def backward(self, L_error):
        if self.conv:
            x_hat = self.x_m / self.x_std
            self.grad_beta = np.sum(L_error,axis=(0,2,3)).reshape(1, x_hat.shape[1], 1, 1)
            self.grad_gamma = np.sum(x_hat * L_error,axis=(0,2,3)).reshape(1, x_hat.shape[1], 1, 1)
            
            gamma = self.gamma.reshape(1, x_hat.shape[1], 1, 1)
            beta = self.beta.reshape(1, x_hat.shape[1], 1, 1)
            Nt = (x_hat.shape[0]*x_hat.shape[2]*x_hat.shape[3])
            grad = (1. / Nt) * gamma * (self.x_var + self.epsilon)**(-1. / 2.) * (Nt * L_error \
                    - np.sum(L_error, axis=(0, 2, 3)).reshape(1, x_hat.shape[1], 1, 1) \
                    - (self.x_m * (self.x_var  + self.epsilon)**(-1.0) *  np.sum(L_error * (self.x_m),axis=(0, 2, 3)).reshape(1, x_hat.shape[1], 1, 1)))
        else:
            x_hat = self.x_m / self.x_std
            self.grad_beta = np.sum(L_error,axis=0)
            self.grad_gamma = np.sum(x_hat * L_error,axis=0)
            BN_fast = (1./x_hat.shape[0]) * self.gamma * (1./self.x_std) * (x_hat.shape[0] * L_error - np.sum(L_error,axis=0))
            BN_fast -= (self.x_m) *  (1./(self.x_std**2)) * np.sum(self.x_m * L_error,axis=0)
            grad = BN_fast


        return grad
    
    def update(self):
        #self.beta -= self.grad_beta 
        #self.gamma -= self.grad_gamma
        '''
            update step (updates weights & momentum)
        '''   
        self.m1b = self.b1*self.m1b + (1-self.b1)*self.grad_beta
        self.m2b = self.b2*self.m2b + (1-self.b2)*self.grad_beta**2
        self.m1 = self.b1*self.m1 + (1-self.b1)*self.grad_gamma
        self.m2 = self.b2*self.m2 + (1-self.b2)*self.grad_gamma**2
        if(self.opt=='RMSprop'):
            self.gamma -= self.eta* self.grad_gamma / (np.sqrt(self.m2) +self.eps)
            self.beta -= self.eta* self.grad_beta / (np.sqrt(self.m2b) +self.eps)
        if (self.opt=='Adam'):
            self.gamma -= self.eta* self.m1 / (np.sqrt(self.m2) +self.eps)
            self.beta -= self.eta* self.m1b / (np.sqrt(self.m2b) +self.eps)
        if(self.opt=='normal'):
            self.gamma -= self.eta* self.grad_gamma
            self.beta -= self.eta* self.grad_beta
        self.count += 1
        
class function_layer:
    '''
    having just the activation for backprop
    '''
    def __init__(self,f):
        self.f = f
        self.w = None #not in use
        
    def forward(self,input_):
        self.activation = input_
        return self.f(input_)

    def backward(self,L_error):
        #print(L_error.shape)
        if len(L_error.shape) > 3:
            #print(L_error.shape)
            return L_error * self.f(self.activation,True).reshape(L_error.shape[0],self.activation.shape[1],L_error.shape[2],L_error.shape[3])
        return L_error * self.f(self.activation,True)

    def update(self):
        '''do nothing !'''
        pass       

class flatten_layer:
    '''
    having just the activation for backprop
    '''
    def __init__(self):
        self.activation = None
        self.w = None #not in use
        
    def forward(self,input_):
        #print(input_.shape,'Input into flatten layer')
        self.activation = input_
        return input_.reshape(input_.shape[0],-1)

    def backward(self,L_error):
        L_error =  L_error.reshape(L_error.shape[0],self.activation.shape[1] ,self.activation.shape[2],self.activation.shape[3])
        #print(L_error.shape,'gradient from flatten layer')
        return L_error

    def update(self):
        '''do nothing !'''
        pass       

### (dropout layer) ## need to specify if not training... w*(1-droptrate)
class dropout_layer:
    '''
        masks activations
            dropout(input,drop)

        parameter:
            drop    : chance for dropping unit

    '''
    def __init__(self,in_dim,drop =.5,training=True): 
        self.training = training
        # dropout mask
        self.drop = drop
        self.mask = np.random.choice([0, 1], size=(in_dim), p=[self.drop, 1-self.drop])


    def forward(self,input_):
        '''
            masks input
        ''' 
        if not self.training: return (1.-self.drop)*input_.T
        return (self.mask*input_)

    def backward(self,L_error):
        '''
            masks backward pass 
        '''
        return self.mask * L_error

    def update(self):   
        '''
            updates mask
        '''   
        self.mask = np.random.choice([0, 1], size=(self.mask.shape[0]), p=[self.drop, 1-self.drop])

    def reset(self):
        '''
            also updates mask
        '''   
        self.update()

class max_pool_layer:
    '''
    max pooling with sizexsize filters
    '''
    def __init__(self,size=2,stride=2):
        
        self.stride = stride
        self.size= size
        self.w = None #not in use
        
    def forward(self,input_):
        '''
            max pooling 
        '''
        self.input_shape = input_.shape
        n_x, d_x, h_x, w_x = self.input_shape
        h_out = (h_x - self.size) / self.stride +1
        w_out = (w_x - self.size) / self.stride +1

        if (not h_out.is_integer() or not w_out.is_integer()):
            print(h_out,w_out)
            raise Exception('decimal pixel output dimension')
        h_out, w_out = int(h_out), int(w_out)
        input_ = input_.reshape(n_x*d_x,1,h_x,w_x)
        self.input_flat = im2col_indices(input_, self.size,  self.size, padding=0, stride=self.stride) 
        self.inds = np.argmax(self.input_flat, axis=0)
        out =  self.input_flat[self.inds, range(self.inds.size)]
        out = out.reshape(h_out, w_out, n_x,d_x)
        out = out.transpose(2,3,0,1)
   
        return out

    def backward(self,L_error):
        '''
                max pool gradient
        ''' 
        n_x, d_x, h_x, w_x = self.input_shape
        zero_block = np.zeros_like(self.input_flat)
        L_error = L_error.transpose(2,3,0,1).ravel()

         
        zero_block[self.inds,range(L_error.size)] = L_error
        grad = zero_block
        grad = col2im_indices(zero_block,(n_x*d_x, 1, h_x, w_x),self.size,self.size , padding=0, stride=self.stride)
        grad = grad.reshape(self.input_shape)
        return grad


    def update(self):
        '''do nothing !'''
        pass       

class avg_pool_layer:
    '''
    max pooling with sizexsize filters
    '''
    def __init__(self,size=2,stride=2):
        
        self.stride = stride
        self.size= size
        self.w = None #not in use
        
    def forward(self,input_):
        '''
            max pooling 
        '''
        self.input_shape = input_.shape
        n_x, d_x, h_x, w_x = self.input_shape
        h_out = (h_x - self.size) / self.stride +1
        w_out = (w_x - self.size) / self.stride +1

        if (not h_out.is_integer() or not w_out.is_integer()):
            print(h_out,w_out)
            raise Exception('decimal pixel output dimension')
        h_out, w_out = int(h_out), int(w_out)
        input_ = input_.reshape(n_x*d_x,1,h_x,w_x)
        self.input_flat = im2col_indices(input_, self.size,  self.size, padding=0, stride=self.stride) 
        #self.inds = np.argmax(self.input_flat, axis=0)
        out =  np.mean(self.input_flat,axis=0)
        out = out.reshape(h_out, w_out, n_x,d_x)
        out = out.transpose(2,3,0,1)
   
        return out

    def backward(self,L_error):
        '''
                max pool gradient
        ''' 
        n_x, d_x, h_x, w_x = self.input_shape
        zero_block = np.zeros_like(self.input_flat)
        L_error = L_error.transpose(2,3,0,1).ravel()

         
        zero_block[:,range(L_error.size)] = (1./self.input_flat.shape[0]) *L_error
        grad = zero_block
        grad = col2im_indices(zero_block,(n_x*d_x, 1, h_x, w_x),self.size,self.size , padding=0, stride=self.stride)
        grad = grad.reshape(self.input_shape)
        return grad


    def update(self):
        '''do nothing !'''
        pass       




########################### MODULE ##################################################################
class module:
    '''
    a module executes a list of layers or other modules, since it can be used like a single layer
        
        * = module([ListOfLayers])

    parameters:
        Layerlist   : list of layers
        erf         : errorfunction
        loss        : last training loss
        
    '''
    def __init__(self,Layerlist):
        self.Layerlist = Layerlist
        self.erf = qef
        

    def infer(self, input_):
        '''
            compute full forward pass
        '''            
        out = input_
        for L in self.Layerlist:
            out = L.forward(out)
        return out

    def forward(self, input_):
        '''
            compute full forward pass
        '''            
        out = input_
        for L in self.Layerlist:
            out = L.forward(out)
            #print('layer output shape : ',out.shape)
        return out

    def train(self,input_,target_):
        '''
            training step
        '''
        outs = self.infer(input_)
        self.loss = self.erf(target_,outs)       
        grad = self.erf(target_,outs,True)        
        for L in self.Layerlist[::-1]:
            #print(grad.shape, 'before grad')
            grad = L.backward(grad)
            #print(grad.shape, 'after grad')

            L.update()
        

    def backward(self,grad):
        '''
            backward step
        '''    
        for L in self.Layerlist[::-1]:
            #print(grad.shape, 'before grad')
            grad = L.backward(grad)
            #print(grad.shape, 'after grad')
        return grad

    def update(self):
        '''
            updating layer weights and momentum
        '''
        for L in self.Layerlist:
            L.update()

########################### MODELS / BLOCKS #######################################################
class dense_block:
    '''
    dense multi layer perceptron model:
        dense_mlp(List_with_layers)

    parameters:
        Layerlist   : list of layers
        erf         : errorfunction
        loss        : last training loss
        
    '''
    def __init__(self,Layerlist,growthrate=12):
        self.Layerlist = Layerlist
        self.growthrate = growthrate
        self.erf = qef
        self.grad = None

    def forward(self, input_):
        '''
            compute full forward pass
            ---with dense looped connections !!!!!!!!!!!!!!!!
        '''            
        out = input_
        self.in_channels = out.shape[1]
        out_list = [out] #list holding all layers activation
        L = self.Layerlist
        for i in range (len (L)):
            #print('before shape ',out.shape)
            #print(out_list[0].shape)
            
            out = np.concatenate(tuple( out_list[:i+1] ),axis=1)
            
            #print('after shape ',out.shape)
            out = L[i].forward(out)
            #print('out ',out.shape)
            out_list += [out]
            #print(out_list[0].shape,out_list[1].shape)
        out_final = np.concatenate(tuple(out_list),axis=1)
        #print(out_final.shape, 'output of dense_block')
        return out_final


    
    def backward(self,L_error):
        L = self.Layerlist
        grad = L_error
        for k in range(len(L)):
            grad = L_error[:,(k)*self.growthrate+1:(k+1)*self.growthrate+1]
            #print(grad.shape, 'gradient part')
            L = L[:(k+1)]

            for i in reversed(range(len(L))):
                grad = L[-(i+1)].backward(grad)
                #print(i,grad.shape,' grad')
                L[-(i+1)].update()
                grad_list = grad[:,:-self.growthrate,:,:]  #grad[:,:-L[-(i+1)].w.shape[0],:,:]
                #print(i,grad_list.shape,'gradlist outer')
                if len(L) > (i+2):
                    for j in (np.arange(len(L)-(i+2))):
                            #print('use the first {} gradients'.format(L[j].w.shape[0]))
                            grad_ = grad_list[:,:self.growthrate,:,:]
                            #print(i,j,grad_.shape,' grad_')
                            grad_list = grad_list[:,self.growthrate:,:,:]
                            #print(i,j,grad_list.shape,'gradlist inner')
                            _ = L[j].backward(grad_)
                            L[j].update()

                grad = grad[:,-self.in_channels:,:,:]
        
        return grad


    def update(self):
        pass

class dense_mlp:
    '''
    dense multi layer perceptron model:
        dense_mlp(List_with_layers)

    parameters:
        Layerlist   : list of layers
        erf         : errorfunction
        loss        : last training loss
        
    '''
    def __init__(self,Layerlist):
        self.Layerlist = Layerlist
        self.erf = qef
        self.grad = None

    def infer(self, input_):
        '''
            compute full forward pass
            ---with dense looped connections !!!!!!!!!!!!!!!!
        '''            
        out = input_
        out_list = [out] #list holding all layers activation
        L = self.Layerlist
        for i in range (len (L)):
            #print('before shape ',out.shape)
            #print(out_list[0].shape)
            out = np.hstack((out_list[j] for j in range(i+1)))
            
            #print('after shape ',out.shape)
            out = L[i].forward(out).T
            #print('out ',out.shape)
            out_list += [out]
            #print(out_list[0].shape,out_list[1].shape)
    
        return out

    def train(self,input_,target_):
        '''
            training step
            ---nested backprop 
        '''
        self.loss = self.erf(target_,self.infer(input_))       
        grad = self.erf(target_,self.infer(input_),True)
        L = self.Layerlist
        #print('round ...........')
        for i in range(len(L)):
            grad = L[-(i+1)].backward(grad)
            L[-(i+1)].update()
            grad_list = grad[:,:-L[-(i+1)].w.shape[0]]
            if len(L) > (i+2):
                for j in (np.arange(len(L)-(i+2))):
                        grad_ = grad_list[:,:L[j].w.shape[0]]
                        grad_list = grad_list[:,L[j].w.shape[0]:]
                        _ = L[j].backward(grad_)
                        L[j].update()

            grad = grad[:,-L[max(-(i+2),-len(L))].w.shape[0]:]  
        self.grad = grad

class dueling_mlp:
    ''' !!!!!!!!!!!!!!!! NOT WORKING YET !!!!!!!!!!!!!!!!
    - not converging so far - think it's about target_updates:
      double Q update :  Q_target = reward + gamma * tagetNet(state+1)[0,argmax(onlineNet(state+1))]
      dueling Q update:  Q_target = reward + gamma * 1/number_of_actions * mean of maybe all Q action values?!
    dueling mlp for Q learning:
        dueling_mlp(LL0,LLA,LLB,model=mlp)
        IN -> LL0 -> [LLV & LLA] => (LLV + (LLA-mean(LLA)))
    parameters:
        LL0         : list of layers for core model
        LLV         : list of layers for value model (shape 1)
        LLA         : list of layers for advantage model (shape actionspace)

    '''
    def __init__(self,LL0,LLV,LLA):
        self.LL0 = LL0
        self.LLV = LLV
        self.LLA = LLA
        self.erf = qef
        

    def infer(self, input_):
        '''
            compute full forward pass over both networks
        '''      
        out0 = input_
        for L in self.LL0:
            out0 = L.forward(out0).T
        outV = out0
        outA = out0
        for L in self.LLA:
            outA = L.forward(outA).T
        for L in self.LLV:
            outV = L.forward(outV).T
       
        outA_ = outA-outA.mean(0)
        outQ = outA_ + outV

        return outQ

    def train(self,input_,target_):
        '''
            training step
            think about the aggregation layer in the end ---> LLout maybe
        '''
        # calculating forward pass
        out0 = input_
        for L in self.LL0:
            out0 = L.forward(out0).T
        outV = out0
        outA = out0
        for L in self.LLA:
            outA = L.forward(outA).T
        for L in self.LLV:
            outV = L.forward(outV).T
        outA_ = outA-outA.mean(0)
        outQ = outA_ + outV
       

        self.TD_loss = np.power((target_-outQ),2) 
        #print(self.TD_loss.shape,'TD loss shape')
        self.loss = self.erf(target_,outQ)
        #print(self.loss.shape,'TD loss shape')
        
        ###################################################
        gradA = self.erf(target_-target_.mean(1)[:,True],outA,True)
        gradV = self.erf(target_.mean(1)[:,True],outV,True)
        # this might be the reason why it's not working. calculate a correct loss please...
        ###################################################
        
        #print(gradV.shape, 'grad V')
        #print(gradA.shape, 'grad A')
        for L in self.LLA[::-1]:
            #print(grad.shape, 'before grad')
            gradA = L.backward(gradA)
            #print(grad.shape, 'after grad')
            L.update()
        for L in self.LLV[::-1]:
            #print(grad.shape, 'before grad')
            gradV = L.backward(gradV)
            #print(grad.shape, 'after grad')
            L.update()
        # 2 gradients arriving in feature extractor LL0
        # update LL0 with the mean of both
        grads = (gradA+gradV)/2.

        for L in self.LL0[::-1]:
            #print(grad.shape, 'before grad')
            gradA = L.backward(grads)
            #print(grad.shape, 'after grad')
            L.update()




########################################## UTILS FROM OTHER PEOPLE #######################
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]



