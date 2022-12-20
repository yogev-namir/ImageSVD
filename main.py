import numpy as np
import pylab
from PIL import Image
import matplotlib.pyplot as plt


def main():
    image = Image.open("C://Users//hadar//Desktop//yossi.JPG")
    rgb = Image.Image.split(image)  # list of 3 RGB images
    uL, sL, vL = SVD(rgb)
    k_list = [5, 10, 20, 30, 40, 50, 100, 200, 300]
    errors = []
    for k in k_list:
        errors.append(calc_error(sL[0], k))
        rgb_low_rank_arr = low_rank_approx(uL, sL, vL, k)
        rgb_low_rank_images = [Image.fromarray(arr.astype('uint8')) for arr in rgb_low_rank_arr]
        low_rank_image = Image.merge('RGB', rgb_low_rank_images)
        low_rank_image.save(str(k) + "yossi.png")
        print("finished f{k}", k)
    plt.plot(k_list, errors, marker='o', linestyle='-')
    plt.xlabel("k-value")
    plt.ylabel("error")
    plt.title("error as a function of k")
    plt.show()


def SVD(rgb):
    uL = []
    sL = []
    vL = []
    for mat in rgb:
        u, s, v = np.linalg.svd(mat, full_matrices=False)
        uL.append(u)
        sL.append(s)
        vL.append(v)
    return uL, sL, vL


def low_rank_approx(uL, sL, vL, k):
    """
    Computes a k-rank approximation of a matrix
    given the components u, s, and v of each of its rgb components,
    """
    Ar = []
    for u, s, v in zip(uL, sL, vL):
        Ar.append(np.zeros((u.shape[0], v.shape[1])))
        for i in range(k):
            Ar[-1] += s[i] * np.outer(u.T[i], v[i])
    return Ar


def calc_error(s, k):
    return sum(pow(s[k + 1:], 2)) / sum(np.power(s, 2))


if __name__ == "__main__":
    main()
