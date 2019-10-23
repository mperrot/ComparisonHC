__all__ = ['ComparisonHC']

import numpy as np
import time

class ComparisonHC:
    """ComparisonHC

    Parameters
    ----------
    linkage : Linkage object
        The linkage used to determine the merging order of the
        clusters.
    
    Attributes
    ----------
    linkage : Linkage object
        The linkage used to determine the merging order of the
        clusters.
    
    clusters : list of (list of examples), len (n_clusters)
        A list containing the initial clusters (list of
        examples). Initialized to the empy list until the fit method is
        called.

    n_clusters : int
        The number of initial clusters. Initialized to 0 until the fit
        method is called.

    dendrogram : numpy array, shape (n_clusters-1, 3)
        An array corresponding to the learned dendrogram. After
        iteration i, dendrogram[i,0] and dendrogram[i,1] are the
        indices of the merged clusters, and dendrogram[i,2] is the
        size of the new cluster. The dendrogram is initialized to all
        0 until the fit method is called.
        
    time_elapsed : float
        The time taken to learn the dendrogram. It includes the time
        taken by the linkage to select the next clusters to merge. It
        only records the time elapsed during the last call to fit.

    Notes
    -----
    The linkage object should exhibit a closest_clusters(clusters)
    method that takes a list of clusters (that is a list of (list of
    examples)) and returns the indices of the two closest clusters
    that should be merged next. This method should be deterministic,
    that is repeated calls to closest_clusters with the same
    parameters should yield the same result.

    """
    def __init__(self,linkage):
        self.linkage = linkage

        self.clusters = []
                
        self.n_clusters = 0

        self.dendrogram = np.zeros((n_clusters-1,4))
                
        self.time_elapsed = 0
            
    def fit(self,clusters):
        """Computes the dendrogram of a list of clusters.

        Parameters
        ----------
        
        clusters : list of (list of examples), len (n_clusters)
            A list containing the initial clusters (list of examples).
        
        Returns
        -------
        self : object

        """
        time_start = time.process_time()
        
        self.clusters = clusters

        self.n_clusters = len(clusters)

        clusters_indices = list(range(self.n_clusters))

        clusters_copy = [[obj for obj in cluster] for cluster in self.clusters]

        for it in range(self.n_clusters-1):
            i,j = self.linkage.closest_clusters(clusters_copy)
            
            if i > j:
                i,j = j,i
            clusters_copy[i].extend(clusters_copy[j])
            del clusters_copy[j]

            self.dendrogram[it,0] = clusters_indices[i]
            self.dendrogram[it,1] = clusters_indices[j]
            self.dendrogram[it,2] = len(clusters_copy[i])

            clusters_indices[i] = self.n_clusters+it
            del clusters_indices[j]
                        
        time_end = time.process_time()
        self.time_elapsed = (time_end-time_start)
                
        return self
