import numpy as np

def planted_model(n_examples_pure=30,levels=3,mu=0.8,delta=0.2,sigma=0.1):
    """This function generate the planted model described in 'Foundations
    of Comparison-Based Hierarchical Clustering'.

    Parameters
    ----------
    n_examples_pure : int
        The number of examples in the pure clusters.

    levels : int
        The number of levels in the hierarchy.

    mu : float
        The expected similarity between pairs belonging to a pure
        cluster.

    delta : float
        The separation between the similarities across consecutive
        levels.

    sigma : float
        The standard deviation of the similarities.

    Returns
    -------
    clusters : list of (list of examples)
        The pure clusters at the bottom of the hierarchy.

    dendrogram_truth : numpy array, shape (n_clusters-1, 3)
        An array corresponding to the true dendrogram. In row i,
        dendrogram[i,0] and dendrogram[i,1] are the indices of the
        merged clusters, and dendrogram[i,2] is the size of the new
        cluster.

    similarities : numpy array, shape (n_examples,n_examples)
        An array containing the noisy similarities between the
        examples.

    """

    n_clusters = 2**levels
    n_examples = n_examples_pure*(n_clusters)

    clusters = [[i*(n_examples_pure)+j for j in range(n_examples_pure)] for i in range(n_clusters)]

    dendrogram_truth = np.zeros((n_clusters-1,3))

    it = 0
    for level in range(levels):
        for i in range(0,n_clusters//(2**level),2):
            dendrogram_truth[it,0] = 2*n_clusters-2**(levels+1-level) + i
            dendrogram_truth[it,1] = 2*n_clusters-2**(levels+1-level) + i + 1
            dendrogram_truth[it,2] = n_examples_pure*(2**(level+1))
            it += 1
        
    similarties_expectation = np.full((n_examples,n_examples),(mu - levels*delta))
    n_examples_level = n_examples
    while n_examples_level>n_examples_pure:
        n_examples_level = n_examples_level//2
        block = 0
        while block<n_examples:
            similarties_expectation[block:block+n_examples_level,:][:,block:block+n_examples_level] += delta
            block = block+n_examples_level

    np.fill_diagonal(similarties_expectation,float("inf"))

    similarities_noise = np.triu(sigma*np.random.normal(size=(n_examples,n_examples)),1)
    similarities_noise = similarities_noise+similarities_noise.transpose()

    similarities = similarties_expectation + similarities_noise
    
    return clusters, dendrogram_truth, similarities
