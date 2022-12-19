import numpy as np
import pylab
from PIL import Image

def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((u.shape[0], v.shape[1]))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

if __name__ == "__main__":
    """
    Test: visualize an r-rank approximation of `lena`
    for increasing values of r
    Requires: scipy, matplotlib
    """

    cat = Image.open(f'cat.jpg')
    gray_cat = cat.convert("L")

    u, s, v = np.linalg.svd(gray_cat, full_matrices=False)
    i = 1
    y = low_rank_approx((u, s, v), r=20)
    Image.fromarray(y.astype('uint8')).show()
