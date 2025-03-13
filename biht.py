import numpy as np

def binary_iterative_hard_thresholding(Phi, y, K, maxiter=3000, htol=0):
    """
    Binary Iterative Hard Thresholding (BIHT) algorithm for 1-bit compressed sensing.
    
    Parameters:
    -----------
    Phi : ndarray
        Sensing matrix of shape (M, N) where M is the number of measurements and 
        N is the signal dimension.
    y : ndarray
        Observed 1-bit measurements of shape (M,).
    K : int
        Sparsity level (number of non-zero elements in the signal).
    maxiter : int, optional
        Maximum number of iterations. Default is 3000.
    htol : int, optional
        Hamming distance tolerance. Default is 0.
        
    Returns:
    --------
    x : ndarray
        Recovered K-sparse signal (normalized).
    stats : dict
        Dictionary containing performance statistics:
        - 'iterations': Number of iterations performed.
        - 'hamming_error': Hamming distance between observed and reconstructed measurements.
    """
    M, N = Phi.shape
    
    # Initialize signal estimate
    x = np.zeros(N)
    hd = float('inf')
    ii = 0
    
    # Measurement function
    def A(input_signal):
        return np.sign(Phi @ input_signal)
    
    while (htol < hd) and (ii < maxiter):
        # Get gradient
        g = Phi.T @ (A(x) - y)
        
        # Step
        a = x - g
        
        # Best K-term approximation (hard thresholding)
        aidx = np.argsort(np.abs(a))[::-1]  # Sort in descending order
        a_thresholded = np.zeros_like(a)
        a_thresholded[aidx[:K]] = a[aidx[:K]]
        
        # Update x
        x = a_thresholded
        
        # Measure Hamming distance to original 1-bit measurements
        hd = np.count_nonzero(y - A(x))
        ii += 1
    
    # Normalize to unit norm
    x = x / np.linalg.norm(x)
    
    # Return results
    stats = {
        'iterations': ii,
        'hamming_error': np.count_nonzero(y - A(x))
    }
    
    return x, stats


# Example usage:
def generate_test_problem(N=2000, M=500, K=15):
    """
    Generate a test problem for 1-bit compressed sensing.
    
    Parameters:
    -----------
    N : int
        Signal dimension.
    M : int
        Number of measurements.
    K : int
        Sparsity level.
        
    Returns:
    --------
    x0 : ndarray
        Original sparse signal.
    Phi : ndarray
        Sensing matrix.
    y : ndarray
        1-bit measurements.
    """
    # Generate a unit K-sparse signal
    x0 = np.zeros(N)
    rp = np.random.permutation(N)
    x0[rp[:K]] = np.random.randn(K)
    x0 = x0 / np.linalg.norm(x0)
    
    # Generate sensing matrix
    Phi = np.random.randn(M, N)
    
    # Get measurements
    y = np.sign(Phi @ x0)
    
    return x0, Phi, y


if __name__ == "__main__":
    # Generate test problem
    N, M, K = 2000, 500, 15
    x0, Phi, y = generate_test_problem(N, M, K)
    
    # Run BIHT
    x_recovered, stats = binary_iterative_hard_thresholding(Phi, y, K)
    
    # Calculate L2 error

    l2_error = np.linalg.norm(x0 - x_recovered) / np.linalg.norm(x0)
    
    # Display results
    print(f"Number of iterations: {stats['iterations']}")
    print(f"L2 error: {l2_error}")
    print(f"Hamming error: {stats['hamming_error']}")