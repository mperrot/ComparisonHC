__all__ = ['Oracle','OraclePassive','OracleComparisons','OracleActive','OracleActiveBudget']

import numpy as np

from abc import ABCMeta,abstractmethod
import time

class Oracle(metaclass=ABCMeta):
    """An abstract oracle that returns quadruplets.

    Parameters
    ----------
    n_examples : int
        The number of examples handled by the oracle.

    seed : int or None
        The seed used to initialize the random number generators. If
        None the current time is used, that is
        int(time.time()). (Default: None).

    Attributes
    ----------
    n_examples : int
        The number of examples handled by the oracle.

    seed : int
        The seed used to initialize the random number generators.

    """
    def __init__(self,n_examples,seed=None):
        self.n_examples = n_examples
                
        if seed is not None:
            self.seed = seed
        else:
            self.seed = int(time.time())
                
    @abstractmethod
    def comparisons(self):
        """Returns all the quadruplets associated with the examples.

        Returns
        -------
        comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
            A reference to a numpy array of shape (n_examples,
            n_examples, n_examples, n_examples) containing values in
            {1,-1,0}. In entry (i,j,k,l), the value 1 indicates that
            the quadruplet (i,j,k,l) is available, the value -1
            indicates that the quadruplet (k,l,i,j) is available, and
            the value 0 indicates that neither of the quadruplets is
            available. This method should be deterministic.

        """
        pass
    
    @abstractmethod
    def comparisons_to_ref(self,k,l):
        """Returns all the quadruplets with respect to the reference of
        examples k,l.

        Returns
        -------
        comparisons_array : numpy array, shape (n_examples, n_examples)
            A reference to a numpy array of shape (n_examples,
            n_examples) containing values in {1,-1,0}. In entry (i,j),
            the value 1 indicates that the quadruplet (i,j,k,l) is
            available, the value -1 indicates that the quadruplet
            (k,l,i,j) is available, and the value 0 indicates that
            neither of the quadruplets is available. This method
            should be deterministic.

        """
        pass
    
    @abstractmethod
    def comparisons_single(self,i,j,k,l):
        """Returns the quadruplet associated with the examples i,j,k,l.

        Returns
        -------
        comparisons_array : int8
            A int8 in {1,-1,0}. The value 1 indicates that the
            quadruplet (i,j,k,l) is available, the value -1 indicates
            that the quadruplet (k,l,i,j) is available, and the value
            0 indicates that neither of the quadruplets is
            available. This method should be deterministic.

        """
        pass
    
class OraclePassive(Oracle):
    """An oracle that returns passively queried quadruplets from standard
    data.

    Parameters
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between
        examples. It should take two numpy arrays of shapes
        (n_examples_1,n_features) and (n_examples_2,n_features) and
        return a distance matrix of shape (n_examples_1,n_examples_2).
    
    proportion_quadruplets : float, optional
        The overall proportion of quadruplets that should be
        generated. (Default: 0.1).

    seed : int or None
        The seed used to initialize the random states. (Default:
        None).

    Attributes
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between examples.
    
    proportion_quadruplets : float
        The overall proportion of quadruplets that should be generated.

    n_examples : int
        The number of examples.

    comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
        A numpy array of shape (n_examples, n_examples, n_examples,
        n_examples) containing values in {1,-1,0}. In entry (i,j,k,l),
        the value 1 indicates that the quadruplet (i,j,k,l) is
        available, the value -1 indicates that the quadruplet
        (k,l,i,j) is available, and the value 0 indicates that neither
        of the quadruplets is available. Initialized to None until one
        of the comparison methods is called.

    seed : int
        The seed used to initialize the random states.

    """    
    def __init__(self,x,metric,proportion_quadruplets=0.1,seed=None):
        self.x = x
        
        self.metric = metric
        
        self.proportion_quadruplets = proportion_quadruplets

        self.comparisons_array = None

        n_examples = x.shape[0]
        super(OraclePassive,self).__init__(n_examples,seed)

    def comparisons(self):
        if self.comparisons_array is None:
            self.comparisons_array = self._get_comparisons()

        return self.comparisons_array
    
    def comparisons_to_ref(self,k,l):
        if self.comparisons_array is None:
            self.comparisons_array = self._get_comparisons()

        return self.comparisons_array[:,:,k,l]
    
    
    def comparisons_single(self,i,j,k,l):
        if self.comparisons_array is None:
            self.comparisons_array = self._get_comparisons()

        return self.comparisons_array[i,j,k,l]
    
    def _get_comparisons(self):
        """Returns all the quadruplets associated with the examples.

        Returns
        -------
        comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
            A reference to a numpy array of shape (n_examples,
            n_examples, n_examples, n_examples) containing values in
            {1,-1,0}. In entry (i,j,k,l), the value 1 indicates that
            the quadruplet (i,j,k,l) is available, the value -1
            indicates that the quadruplet (k,l,i,j) is available, and
            the value 0 indicates that neither of the quadruplets is
            available. This method should be deterministic.

        """
        random_state = np.random.RandomState(self.seed)
                
        similarities = self.metric(self.x,self.x)
            
        comparisons_array = np.zeros((self.n_examples,self.n_examples,
                          self.n_examples,self.n_examples),dtype='int8')

        # This is to take into account the symmetry effect that makes us query each quadruplet twice
        proportion_effective = (1-np.sqrt(4-4*self.proportion_quadruplets)/2)
    
        for i in range(self.n_examples):
            for j in range(i+1,self.n_examples):
                selector = np.triu(random_state.rand(self.n_examples,self.n_examples),1)
                selector = (selector + selector.transpose())<proportion_effective

                comparisons_array[i,j,:,:] = np.where(np.logical_and(selector,similarities[i,j] > similarities),1,0) + np.where(np.logical_and(selector,similarities[i,j] < similarities),-1,0)
                comparisons_array[j,i,:,:] = comparisons_array[i,j,:,:]

        comparisons_array -= comparisons_array.transpose()
        comparisons_array = np.clip(comparisons_array,-1,1)
    
        return comparisons_array

class OracleComparisons(Oracle):
    """An oracle that returns quadruplets from a precomputed numpy array.

    Parameters
    ----------
    comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
        A numpy array of shape (n_examples, n_examples, n_examples,
        n_examples) containing values in {1,-1,0}. In entry (i,j,k,l),
        the value 1 indicates that the quadruplet (i,j,k,l) is
        available, the value -1 indicates that the quadruplet
        (k,l,i,j) is available, and the value 0 indicates that neither
        of the quadruplets is available.

    Attributes
    ----------
    n_examples : int
        The number of examples.

    comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
        A numpy array of shape (n_examples, n_examples, n_examples,
        n_examples) containing values in {1,-1,0}. In entry (i,j,k,l),
        the value 1 indicates that the quadruplet (i,j,k,l) is
        available, the value -1 indicates that the quadruplet
        (k,l,i,j) is available, and the value 0 indicates that neither
        of the quadruplets is available.

    """    
    def __init__(self,comparisons_array):
        self.comparisons_array = comparisons_array
        
        n_examples = comparisons_array.shape[0]
        super(OracleComparisons,self).__init__(n_examples)

    def comparisons(self):
        return self.comparisons_array
    
    def comparisons_to_ref(self,k,l):
        return self.comparisons_array[:,:,k,l]
    
    
    def comparisons_single(self,i,j,k,l):
        return self.comparisons_array[i,j,k,l]
    
class OracleActive(Oracle):
    """An oracle that returns actively queried quadruplets from standard
    data.

    Parameters
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between
        examples. It should take two numpy arrays of shapes
        (n_examples_1,n_features) and (n_examples_2,n_features) and
        return a distance matrix of shape (n_examples_1,n_examples_2).
    
    seed : int or None
        The seed used to initialize the random states. (Default:
        None).

    Attributes
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between examples.
    
    n_examples : int
        The number of examples.

    similarities : numpy array, shape (n_examples,n_examples)
        A numpy array containing the similarities between all the
        examples. Initialized to None until the first call to one of
        the comparisons method.

    seed : int
        The seed used to initialize the random states.

    """    
    def __init__(self,x,metric,seed=None):
        self.x = x
        n_examples = x.shape[0]
        
        self.metric = metric

        self.similarities = None
        
        super(OracleActive,self).__init__(n_examples,seed)

    def comparisons(self):
        raise NotImplemented("Querying all the quadruplets with an active oracle is prohibited.")
    
    def comparisons_to_ref(self,k,l):
        if self.similarities is None:
            self.similarities = self.metric(self.x,self.x)

        comparisons_array = (self.similarities > self.similarities[k,l])*1 - (self.similarities < self.similarities[k,l])*1
                    
        return comparisons_array
    
    def comparisons_single(self,i,j,k,l):
        if self.similarities is None:
            self.similarities = self.metric(self.x,self.x)

        comparisons_array = (self.similarities[i,j] > self.similarities[k,l])*1 - (self.similarities[i,j] < self.similarities[k,l])*1
                    
        return comparisons_array
    

class OracleActiveBudget(OracleActive):
    """An oracle that returns actively queried quadruplets from standard
    data within a given budget.

    This oracle queries quadruplets with respect to reference pairs
    only and can only query a limited number of quadruplets
    (controlled by a proportion of quadruplets that can be queried).

    Parameters
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between
        examples. It should take two numpy arrays of shapes
        (n_examples_1,n_features) and (n_examples_2,n_features) and
        return a distance matrix of shape (n_examples_1,n_examples_2).
    
    proportion_quadruplets : float, optional
        The overall proportion of quadruplets that should be
        generated. (Default: 0.1).

    seed : int or None
        The seed used to initialize the random states. (Default:
        None).

    Attributes
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between examples.
    
    proportion_quadruplets : float
        The overall proportion of quadruplets that should be generated.

    budget : int
        The maximum number of pairs that can be queried.

    n_examples : int
        The number of examples.

    references : list of (int,int)
        A list containing the reference pairs that have already been
        queried.

    similarities : numpy array, shape (n_examples,n_examples)
        A numpy array containing the similarities between all the
        examples. Initialized to None until the first call to one of
        the comparisons method.

    seed : int
        The seed used to initialize the random states.

    """    
    def __init__(self,x,metric,proportion_quadruplets=0.1,seed=None):
        super(OracleActiveBudget,self).__init__(x,metric,seed)
                
        self.proportion_quadruplets = proportion_quadruplets

        # Compute the number of quadruplets for n_examples (excluding obvious quadruplets)
        # First substraction is to remove obvious quadruplets of the form i,i,k,l and i,i,l,k and k,l,i,i and l,k,i,i
        # Second substraction is to remove obvious quadruplets of the form i,i,j,j
        # Third substraction is to remove obvious quadruplets of the form i,j,i,j and j,i,i,j and i,j,j,i and j,i,j,i
        # Divided by 8 since each quadruplet has 8 counterparts with the same meaning
        effective_quadruplets = (self.n_examples**4 - 2*self.n_examples*self.n_examples*(self.n_examples-1) - self.n_examples**2 - 2*self.n_examples*(self.n_examples-1))/8
        # Compute the number of effective quadruplets for a given reference pair, the -1 is to account for the case i,j,i,j
        effective_quadruplets_ref = self.n_examples*(self.n_examples-1)/2 - 1
        # After rep repetitions the effective_quadruplets_ref of the new pair is effective_quadruplets_ref-rep+1 because of the symmetry
        # Hence we have to sovle rep*(effective_quadruplets_ref+effective_quadruplets_ref-rep+1)/2 <= effective_quadrupletss*proportion_quadruplets
        if self.proportion_quadruplets >= 1:
            self.budget = int(self.n_examples*(self.n_examples-1)/2)
        else:
            self.budget = int(effective_quadruplets_ref + 1/2 - np.sqrt((2*effective_quadruplets_ref + 1)**2 - 8*effective_quadruplets*self.proportion_quadruplets)/2)

        self.references = []

    def comparisons_to_ref(self,k,l):
        """Returns all the quadruplets with respect to the reference of
        examples k,l.

        Returns
        -------
        comparisons_array : numpy array, shape (n_examples, n_examples)
            A reference to a numpy array of shape (n_examples,
            n_examples) containing values in {1,-1,0}. In entry (i,j),
            the value 1 indicates that the quadruplet (i,j,k,l) is
            available, the value -1 indicates that the quadruplet
            (k,l,i,j) is available, and the value 0 indicates that
            neither of the quadruplets is available. This method is
            deterministic. This array is None if the budget has been
            reached.

        """
        if self.similarities is None:
            self.similarities = self.metric(self.x,self.x)

        if k > l:
            k,l = l,k

        if (k,l) in self.references:
            comparisons_array = (self.similarities > self.similarities[k,l])*1 - (self.similarities < self.similarities[k,l])*1
        elif self.budget > len(self.references):
            comparisons_array = (self.similarities > self.similarities[k,l])*1 - (self.similarities < self.similarities[k,l])*1
            self.references.append((k,l))
        else:
            comparisons_array = None
            
        return comparisons_array
    
    def comparisons_single(self,i,j,k,l):
        raise NotImplemented("Querying a single quadruplet with a budgeted active oracle is prohibited.")
