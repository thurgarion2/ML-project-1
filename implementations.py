import numpy as np
from functools import partial
import csv

#####helpers
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - (tx@w) 
    return 0.5*np.mean(e**2)

def compute_rmse(y, tx, w):
    """compute the loss by mse."""
    mse = compute_mse(y, tx, w)
    return np.sqrt(2*mse)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx@w 
    N = len(y)
    return -tx.T@e/N



#########optimisation

def least_square_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_gradient(y,tx,w)
        loss=compute_mse(y,tx,w)
        w=w-gamma*grad
   

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws=[initial_w]
    losses=[]
    w=initial_w
    for n_iter in range(max_iters):
      for minibatch_y, minibatch_tx in batch_iter(y, tx, 1, num_batches=1, shuffle=True):
        loss=compute_mse(y,tx, w)
        stoch_grad=compute_gradient(minibatch_y,minibatch_tx,w)
        w=w-gamma*stoch_grad
        ws.append(w)
        losses.append(loss)
    return ws[-1], losses[-1] 




def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # returns optimal weights, MSE
    # ***************************************************
    target = tx.T@y
    x_t_x = tx.T@tx
    w = np.linalg.solve(x_t_x,target)
    return w, compute_mse(y, tx, w)



def ridge_regression(y, tx, lambda_):
  #compute the ridge regression of y = tx.w
  #return weights and loss (in MSE)
  N=tx.shape[0]
  lambda_prime = 2*N*lambda_
  G=np.transpose(tx).dot(tx) + lambda_prime*np.eye(tx.shape[1])
  w = np.linalg.solve(G, np.transpose(tx).dot(y))
  loss= compute_mse(y,tx,w)  + lambda_*((np.linalg.norm(w))**2)
  return w, loss


def logistic_function(x):
  return 1/(1+np.exp(-x))

def calculate_loss(y, tx, w):
  """compute the loss: negative log likelihood."""
  prediction = logistic_function(tx@w) 

  ####log of 0 does not exist
  prediction[prediction == 0]= 0.0000000000001
  prediction[prediction == 1]= 0.9999999999999  
  loss_per_prediction = y*np.log(prediction) + (1-y)*np.log(1-prediction)
  return -loss_per_prediction.sum()

def calculate_gradient(y, tx, w):
  """compute the gradient of loss."""
  # ***************************************************
  prediction=logistic_function(tx@w)
  gradient = tx.T.dot(prediction - y)
  return gradient

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss=calculate_loss(y, tx, w)
    grad=calculate_gradient(y,tx,w)
    w= w-gamma*grad
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    losses = [] 
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
       
    return w, loss




def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    # ***************************************************
    loss=calculate_loss(y,tx,w) + lambda_*((np.linalg.norm(w))**2) 
    grad= calculate_gradient(y, tx, w) + 2*lambda_*w
    return loss, grad
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w =w -grad*gamma   
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  
  losses = []
  w=initial_w
  # start the logistic regression
  for iter in range(max_iters):
    # get loss and update w.
    loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    losses.append(loss)
  
  return w, losses[-1]



###########################################################################
##########################Helpers##########################################

def load_columns_name(path):
  with open(path) as f:
    cols = f.readline()
    cols = cols.split(',')[2:]
    cols[-1] = cols[-1][:-1]
    return cols

def load_features(path):
    x = np.genfromtxt(path, delimiter=",", skip_header=1)
    return x[:, 2:]

def load_labels(path):
    y = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    return yb

def load_ids(path):
  return np.genfromtxt(path, delimiter=",", skip_header=1, usecols=0)

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
###represent a dataset, a we need to train a model and uderstand the features
class Dataset:

  def __init__(self):
    self.x = None
    self.y = None
    self.feature_name = None
  
  def with_values(self, x, y, feature_name):
    self.x = x
    self.y = y
    self.feature_name = feature_name
    return self

  def from_path(self, path):
    self.x = load_features(path)
    self.y = load_labels(path)
    self.feature_name = load_columns_name(path)
    return self
  
  def __repr__(self):
    return f'features: {self.feature_name}'

  def split(self, ratio):
    shuffled = np.random.permutation(len(self.y))

    split_idx = int(ratio*len(self.y))
    train, test = shuffled[:split_idx], shuffled[split_idx:]

    train_ds =  Dataset().with_values(self.x[train], self.y[train], self.feature_name)
    test_ds =  Dataset().with_values(self.x[test], self.y[test], self.feature_name)

    return train_ds, test_ds

  def plot_box(self, feature_index, ax):
    ax.set_title(self.feature_name[feature_index])
    ax.boxplot(self.x[:,feature_index])
  
  def plot_hist(self, feature_index, ax):
    ax.set_title(self.feature_name[feature_index])
    ax.hist(self.x[:,feature_index], bins = 40)
  
  def scatter_plot(self, feature_1 : int, feature_2 : int, ax):
    ax.set_xlabel(self.feature_name[feature_1])
    ax.set_ylabel(self.feature_name[feature_2])
    

    ax.scatter(self.x[:,feature_1], self.x[:,feature_2])

def predict_label(y, threshold=0):
  predictions = np.ones((len(y)))
  predictions[y<threshold] = -1
  return predictions

def accuracy(y, y_pred):
  return np.mean(y == y_pred)

#we define the function eval_model that allow us to evaluate more easily our model in the future giving back both accuacies and losses
#once we specify the model, the optimization function and the type of preprocessing we want to use
def eval_model(train_dataset, 
               test_dataset,
               model, 
               preprocess,
               opti_function,
               print_result = True):
  
  train_dataset = preprocess(train_dataset)
  test_dataset = preprocess(test_dataset)
  
  w, _ = opti_function(train_dataset.y, train_dataset.x)

  y_train, loss_train = model(w, train_dataset.x, train_dataset.y)
  y_test, loss_test = model(w, test_dataset.x, test_dataset.y)


  acc_train = accuracy(train_dataset.y, y_train)
  acc_test = accuracy(test_dataset.y, y_test)

  if print_result:
    print('test results')
    print('###################')
    print(f'train error {loss_train}')
    print(f'test accuracy {acc_test}')
    print('###################')
  
  return w, loss_train, loss_test, acc_train, acc_test

def expand_features(dataset, expansions):
  feature_name = ['bias'] + dataset.feature_name
  features = [np.ones((dataset.x.shape[0],1)), dataset.x] 

  for expansion in expansions:
    f_name, f = expansion(dataset)
    feature_name = feature_name + f_name
    features.append(f)
  
  return Dataset().with_values(np.concatenate(features, axis=1),
                               dataset.y,
                               feature_name)

def ploy_expand(power, ds):
  f_name = [ f'{f}^{power}' for f in ds.feature_name]
  return f_name, ds.x**power

def sin_expand(ds):
   f_name = [ f'{f}-sin' for f in ds.feature_name]
   return f_name, np.sin(ds.x)

def cos_expand(ds):
  f_name = [ f'{f}-cos' for f in ds.feature_name]
  return f_name, np.cos(ds.x)

def log_expand(ds, min_features):
  f_name = [f'{f}-log' for f in ds.feature_name]
  x = ds.x-min_features+1
  x[x<1] = 1

  return f_name, np.log(x)



def augment_with(expansions):
  return lambda dataset: expand_features(dataset, expansions)



