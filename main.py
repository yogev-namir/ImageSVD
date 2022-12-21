import numpy as np
import pylab
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# -------------------- question 7 -------------------------
from knn import KNN


def Q7():
    # train_files = ["data_batch_1", "data_batch_2",
    #                "data_batch_3", "data_batch_4", "data_batch_5"]
    file1 = "data_batch_1"
    file2 = "test_batch"
    my_dict_test = unpickle(file2)
    img_test = my_dict_test[b'data']
    grayImg_arr_test = grayScale(img_test)
    my_dict_train = unpickle(file1)
    img_train = my_dict_train[b'data']
    labels = my_dict_train[b'labels']
    grayImg_arr_train = grayScale(img_train)
    u, sigma, v = PCA(grayImg_arr_train)
    s_list = [20, 50, 100, 200, 400, 1024]
    k_list = [15, 115, 505, 2005, 5005]
    errors = {k: [] for k in k_list}

    for s in s_list:
        model = KNN(1)  # default k
        x_train = (transform(grayImg_arr_train.T, u, s)).T
        x_test = (transform(grayImg_arr_test.T, u, s)).T
        for k in k_list:
            model.fit(x_train, labels, k)
            prediction = np.array(model.predict(x_test[:50, :]))
            error_rate = errorRate(prediction, my_dict_test[b'labels'][:50])
            errors[k].append(round(error_rate, 3))  # for model in models:

    # for k in k_list:
    #     model = KNN(k)
    #     model.fit(grayImg_arr_train, my_dict_train[b'labels'], k)
    #     prediction = model.predict(grayImg_arr_test[:50, :])
    #     error_rate = errorRate(prediction, my_dict_test[b'labels'][:50])
    #     errors[k].append(round(error_rate, 3))

    for k in k_list:
        data = errors[k]
        plt.plot(data, label=f'k={k}', marker='o')
    plt.xlabel('s')
    plt.ylabel('error rate')
    plt.legend()
    plt.show()
    error_df = pd.DataFrame.from_dict(errors).transpose()
    s = [20, 50, 100, 200, 400, 'No PCA']
    error_df.index = k_list
    error_df.columns = s
    print(error_df)


def transform(mat, u, s):
    # projection_mat = np.matmul(u[:, :s], u[:, :s].T)
    return np.matmul(u[:, :s].T, mat)


def explained_variance_ratio(sigma):
    T = [1, 2, 5, 10, 20, 40, 100, 250]
    ratio = [np.sum(sigma[:t] ** 2) / np.sum(sigma ** 2) for t in T]
    plt.plot(T, ratio, marker='o')
    plt.xticks(T)
    plt.show()


def grayScale(img):
    grayImg_arr = np.zeros((10000, 1024))
    for i, img in enumerate(img):
        single_img = np.array(img)
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(single_img_reshaped.astype('uint8'))
        gray_image = image.convert("L")
        grayimg = np.array(gray_image).flatten()
        grayImg_arr[i] = grayimg
    return grayImg_arr


def errorRate(y_pred, y_true):
    return (1 / len(y_true)) * np.sum(y_pred != y_true)


def PCA(mat):
    variance_mat = (mat - np.average(mat, axis=0)).T
    u, s, v = np.linalg.svd(variance_mat)
    return u, s, v


def unpickle(file):
    with open(file, 'rb') as fo:
        q7_dict = pickle.load(fo, encoding='bytes')
    return q7_dict


# ------------------------- question 6
def Q6():
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


# -----------------------------

def main():
    Q7()


if __name__ == "__main__":
    main()
