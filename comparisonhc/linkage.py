__all__ = ['OrdinalLinkage','OrdinalLinkageKernel','OrdinalLinkageAverage','OrdinalLinkageSingle','OrdinalLinkageComplete']

import numpy as np

from abc import ABCMeta,abstractmethod
import time,itertools,random

class OrdinalLinkage(metaclass=ABCMeta):
    """An abstract ordinal linkage that controls the merging order in
    hierarchical clustering.

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the quadruplets.

    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the quadruplets.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.


    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage.

    """
    def __init__(self,oracle):
        self.oracle = oracle
        
        self.time_elapsed = 0
                        
    @abstractmethod
    def closest_clusters(clusters):
        """Returns the indices of the two clusters that are closest to each
        other in the list.

        Given a list of clusters, this method should be deterministic.

        Parameters
        ----------
        clusters : list of (list of examples)
            A list containing the clusters (list of examples).
        
        Returns
        -------
        i : int
            The index of the first of the two closest clusters.

        j : int
            The index of the second of the two closest clusters.
        
        """
        pass

    
class OrdinalLinkageKernel(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.
    
    This method first computes kernel similarities between the examples
    before using an average linkage scheme.

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the quadruplets.

    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the quadruplets.

    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the
        examples. Initialized to None until the first call to
        closest_clusters.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.

    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage, that is it
        does not exhibit a method comparisons_to_ref(k,l) and an attribute
        n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons_to_ref(k,l) that returns a numpy array of shape
    (n_examples, n_examples) containing values in {1,-1,0}. In entry
    (i,j), the value 1 indicates that the quadruplet (i,j,k,l) is
    available, the value -1 indicates that the quadruplet (k,l,i,j) is
    available, and the value 0 indicates that neither of the
    quadruplets is available. This method should be deterministic.

    The oracle should also exhibit an attribute n_examples counting the
    number of examples it handles.

    For an active oracle, a call to comparisons_to_ref(k,l) for a new pair
    (k,l) should return None when the budget of the oracle is reached.

    """    
    def __init__(self,oracle):
        if not (hasattr(oracle,"n_examples") and hasattr(oracle,"comparisons_to_ref")
                and callable(getattr(oracle,"comparisons_to_ref"))):
            raise ValueError("Incompatible oracle, callable 'comparisons_to_ref' or attribute 'n_examples' missing.")

        self.kernel = None
        
        super(OrdinalLinkageKernel,self).__init__(oracle)

    def closest_clusters(clusters):
        """Returns the indices of the two clusters that are closest to each
        other in the list.

        Given a list of clusters, this method is deterministic.

        Parameters
        ----------
        clusters : list of (list of examples)
            A list containing at least two clusters.
        
        Returns
        -------
        i : int
            The index of the first of the two closest clusters.

        j : int
            The index of the second of the two closest clusters.
        
        """
        time_start = time.process_time()

        if self.kernel is None:
            self.kernel = self._get_kernel()
                        
        n_clusters = len(clusters)

        i,j = None,None
        
        score_best = -float("inf")

        for p in range(n_clusters):
            for q in range(p+1,n_clusters):
                kernel_pq = self.kernel[clusters[p],:][:,clusters[q]]

                score = np.mean(kernel_pq)
                            
                if score > score_best:
                    i,j = p,q
                    score_best = score

        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)
        
        return  i,j

    def _get_kernel(self):
        """Returns a kernel matrix representing the similarities between all
        the examples and the number of examples handled by the oracle.

        Returns a numpy array of shape (n_examples, n_examples)
        containing the similarties between all the examples handled by
        the oracle.

        Returns
        -------
        kernel : numpy array, shape (n_examples,n_examples)
            A nummpy array of similarities between the examples.

        Notes
        -----
        This method should only be called once as it is not
        deterministic.

        """
        kernel = np.zeros((self.oracle.n_examples,self.oracle.n_examples))

        combs = list(itertools.combinations(range(self.oracle.n_examples),2))
        random.shuffle(combs)
        
        for k,l in combs:
            comparisons = self.oracle.comparisons_to_ref(k,l)

            # Check whether the budget is exhausted for an active oracle
            if comparisons is None:
                break
            
            for i in range(self.oracle.n_examples):
                kernel[i,i+1:] += (comparisons[i+1:,:].astype(int)
                                   @ comparisons[i,:].astype(int))
                
        kernel += kernel.transpose()

        return kernel
    
class OrdinalLinkageAverage(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.
    
    This method directly use the quadruplets in an average linkage
    scheme.

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the quadruplets.

    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the quadruplets.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.

    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage, that is it
        does not exhibit a method comparisons() and an attribute
        n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons() that returns a numpy array of shape
    (n_examples, n_examples, n_examples, n_examples) containing values in
    {1,-1,0}. In entry (i,j,k,l), the value 1 indicates that the
    quadruplet (i,j,k,l) is available, the value -1 indicates that the
    quadruplet (k,l,i,j) is available, and the value 0 indicates that
    neither of the quadruplets is available. This method should be
    deterministic. This numpy array is not modified by this class to
    ensure that it can be passed by reference.

    The oracle should also exhibit an attribute n_examples counting the
    number of examples it handles.

    """    
    def __init__(self,oracle):
        if not (hasattr(oracle,"n_examples") and hasattr(oracle,"comparisons")
                and callable(getattr(oracle,"comparisons"))):
            raise ValueError("Incompatible oracle, callable 'comparisons' or attribute 'n_examples' missing.")

        super(OrdinalLinkageAverage,self).__init__(oracle)

    def closest_clusters(clusters):
        """Returns the indices of the two clusters that are closest to each
        other in the list.

        Given a list of clusters, this method is deterministic.

        Parameters
        ----------
        clusters : list of (list of examples)
            A list containing at least two clusters.
        
        Returns
        -------
        i : int
            The index of the first of the two closest clusters.

        j : int
            The index of the second of the two closest clusters.
        
        """
        time_start = time.process_time()
        
        n_clusters = len(clusters)

        comparisons = self.oracle.comparisons()

        i,j = None,None
        
        score_best = -1

        # Prepare the normalization array
        # This is the divisor for each entry in the sum.
        # It depends on the cluster of each example.
        normalization = np.zeros((1,1,self.oracle.n_examples,self.oracle.n_examples))
        for r in range(n_clusters):
            normalization[0,0,[np.array(clusters[r]).reshape(-1,1)],
                          np.isin(np.arange(n),clusters[r],invert=True)] = 1/len(clusters[r])
            
        for s in range(n_clusters):
            normalization[0,0,:,clusters[s]] /= len(clusters[s])
                            
        for p in range(n_clusters):
            clusters_p = clusters[p]
            
            comparisons_p = comparisons[clusters_p,:,:,:]

            n_examples_p = len(clusters_p)

            for q in range(p+1,n_clusters):
                score = 0
                                
                clusters_q = clusters[q]
                
                comparisons_pq = comparisons_p[:,clusters_q,:,:]
                
                n_examples_pq = n_examples_p*len(clusters_q)

                # Divide each entry in the matrix by the normalization and sum everything
                score = np.sum(comparisons_pq*normalization)/(n_clusters*(n_clusters-1)*n_examples_pq)

                if score > score_best:
                    i,j = p,q
                    score_best = score
       
        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)
        
        return  i,j
    
class OrdinalLinkageSingle(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.
    
    This method directly use the quadruplets in a single linkage
    scheme.

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the quadruplets.

    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the quadruplets.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.

    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage, that is it
        does not exhibit a method comparisons_single(i,j,k,l) and an
        attribute n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons_single(i,j,k,l) that returns a value in
    {1,-1,0}. In entry (i,j), the value 1 indicates that the
    quadruplet (i,j,k,l) is available, the value -1 indicates that the
    quadruplet (k,l,i,j) is available, and the value 0 indicates that
    neither of the quadruplets is available. This method should be
    deterministic.

    """    
    def __init__(self,oracle):
        if not (hasattr(oracle,"comparisons_single")
                and callable(getattr(oracle,"comparisons_single"))):
            raise ValueError("Incompatible oracle, callable 'comparisons_single' missing.")

        super(OridnalLinkageSingle,self).__init__(oracle)

    def closest_clusters(clusters):
        """Returns the indices of the two clusters that are closest to each
        other in the list.

        Given a list of clusters, this method is deterministic.

        Parameters
        ----------
        clusters : list of (list of examples)
            A list containing at least two clusters.
        
        Returns
        -------
        i : int
            The index of the first of the two closest clusters.

        j : int
            The index of the second of the two closest clusters.
        
        """
        time_start = time.process_time()
        
        n_clusters = len(clusters)

        i,j = None,None

        for p in range(n_clusters):
            for q in range(p+1,n_clusters):
                if self._is_closer(clusters[p],clusters[q],clusters[i],clusters[j]):
                    i,j = p,q
       
        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)
        
        return  i,j

    def _is_closer(cluster_p,cluster_q,cluster_i,cluster_j):
        """Returns a Boolean indicating whether the clusters cluster_p and
        cluster_q are closer to each other than the clusters cluster_i
        and cluster_j.

        Parameters
        ----------
        cluster_p : list of examples
            The first cluster of the first pair.

        cluster_q : list of examples
            The second cluster of the first pair.

        cluster_i : list of examples
            The first cluster of the second pair.

        cluster_j : list of examples
            The second cluster of the second pair.
        
        Returns
        -------
        : Boolean
            Whether cluster_p and cluster_q are closer to each other
            than cluster_i and cluster_j.

        """
        cluster_p_ref = cluster_p[0]
        cluster_q_ref = cluster_q[0]

        for k in cluster_p:
            for l in cluster_q:
                if self.oracle.comparisons_single(k,l,cluster_p_ref,cluster_q_ref) == 1:
                    cluster_p_ref = k
                    cluster_q_ref = l
            
        cluster_i_ref = cluster_i[0]
        cluster_j_ref = cluster_j[0]
        
        for k in cluster_i:
            for l in cluster_j:
                if self.oracle.comparisons_single(k,l,cluster_i_ref,cluster_j_ref) == 1:
                    cluster_i_ref = k
                    cluster_j_ref = l

        return self.oracle.comparisons_single(cluster_p_ref,cluster_q_ref,cluster_i_ref,cluster_j_ref) == 1

class OrdinalLinkageComplete(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.
    
    This method directly use the quadruplets in a complete linkage
    scheme.

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the quadruplets.

    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the quadruplets.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.

    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage, that is it
        does not exhibit a method comparisons_single(i,j,k,l) and an
        attribute n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons_single(i,j,k,l) that returns a value in
    {1,-1,0}. In entry (i,j), the value 1 indicates that the
    quadruplet (i,j,k,l) is available, the value -1 indicates that the
    quadruplet (k,l,i,j) is available, and the value 0 indicates that
    neither of the quadruplets is available. This method should be
    deterministic.

    """    
    def __init__(self,oracle):
        if not (hasattr(oracle,"comparisons_single")
                and callable(getattr(oracle,"comparisons_single"))):
            raise ValueError("Incompatible oracle, callable 'comparisons_single' missing.")

        super(OridnalLinkageComplete,self).__init__(oracle)

    def closest_clusters(clusters):
        """Returns the indices of the two clusters that are closest to each
        other in the list.

        Given a list of clusters, this method is deterministic.

        Parameters
        ----------
        clusters : list of (list of examples)
            A list containing at least two clusters.
        
        Returns
        -------
        i : int
            The index of the first of the two closest clusters.

        j : int
            The index of the second of the two closest clusters.
        
        """
        time_start = time.process_time()
        
        n_clusters = len(clusters)

        i,j = None,None

        for p in range(n_clusters):
            for q in range(p+1,n_clusters):
                if self._is_closer(clusters[p],clusters[q],clusters[i],clusters[j]):
                    i,j = p,q
       
        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)
        
        return  i,j

    def _is_closer(cluster_p,cluster_q,cluster_i,cluster_j):
        """Returns a Boolean indicating whether the clusters cluster_p and
        cluster_q are closer to each other than the clusters cluster_i
        and cluster_j.

        Parameters
        ----------
        cluster_p : list of examples
            The first cluster of the first pair.

        cluster_q : list of examples
            The second cluster of the first pair.

        cluster_i : list of examples
            The first cluster of the second pair.

        cluster_j : list of examples
            The second cluster of the second pair.
        
        Returns
        -------
        : Boolean
            Whether cluster_p and cluster_q are closer to each other
            than cluster_i and cluster_j.

        """
        cluster_p_ref = cluster_p[0]
        cluster_q_ref = cluster_q[0]

        for k in cluster_p:
            for l in cluster_q:
                if self.oracle.comparisons_single(cluster_p_ref,cluster_q_ref,k,l) == 1:
                    cluster_p_ref = k
                    cluster_q_ref = l
            
        cluster_i_ref = cluster_i[0]
        cluster_j_ref = cluster_j[0]
        
        for k in cluster_i:
            for l in cluster_j:
                if self.oracle.comparisons_single(cluster_i_ref,cluster_j_ref,k,l) == 1:
                    cluster_i_ref = k
                    cluster_j_ref = l

        return self.oracle.comparisons_single(cluster_p_ref,cluster_q_ref,cluster_i_ref,cluster_j_ref) == 1
