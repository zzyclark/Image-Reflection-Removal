import cv2
import numpy as np
import numpy.linalg as la
import os

files = [f for f in os.listdir('.') if os.path.isfile(f)]
imgs= []

# read input
for f in files:
    if 'png' in f and 'reflection' not in f and 'background' not in f:
        imgs.append(cv2.imread(f))
    elif 'noreflection' in f:
        noreflection = cv2.imread(f)

# generate matrix for robust pca
num = len(imgs)
h, w = imgs[0].shape[:2]
imgresh = np.zeros((h*w*3, num))
for index in range(num):
    imgresh[:, index] = np.reshape(imgs[index], (h*w*3))

def shrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

def norm_p(M, p):
    return np.sum(np.power(M, p))

def svd_threshold(M, tau):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    trim_S = shrink(S, tau)
    return np.dot(U, np.dot(np.diag(trim_S), V))

def compute_mu(D):
    return np.prod(D.shape) / (4 * norm_p(D, 2))

def Jfunction(X,lam):
    X = np.sign(X)
    U, S, V = la.svd(X, full_matrices=False)
    s_norm = np.max(S)
    i_norm = 1/lam*np.max(np.absolute(X))
    return X/max(s_norm,i_norm)

def robust_pca(D, max_iter=1000):
    iter = 0
    err = np.Inf
    S_temp = Sk = np.zeros(D.shape)
    L_temp = Lk = np.zeros(D.shape)

    mu = compute_mu(D)
    lmbda = 1 / np.sqrt(np.max(D.shape))
    Yk = Jfunction(D, lmbda)

    tol = 1E-7
    tol_primal = 1E-5

    while (err > tol) and iter < max_iter:
        primal_converge = False
        primal_iter = 0
        print 'iter 1: ', iter
        Lk = svd_threshold(D - Sk + Yk/mu, 1/mu)
        Sk = shrink(D - Lk + Yk/mu, lmbda/mu)

        while (not primal_converge) and primal_iter < 100:
           print 'iter 2: ', primal_iter
           Lk = svd_threshold(D - Sk + Yk/mu, 1/mu)
           Sk = shrink(D - Lk + Yk/mu, lmbda/mu)
           print 'L norm: ', la.norm(L_temp - Lk)
           print 'S norm: ', la.norm(S_temp - Sk)
           if la.norm(L_temp - Lk) < tol_primal and la.norm(S_temp - Sk) < tol_primal:
               primal_converge = True
           S_temp = Sk
           L_temp = Lk
           primal_iter = primal_iter + 1

        Yk = Yk + mu * (D - Lk - Sk)
        mu = mu * 1.5
        err = la.norm(D - Lk - Sk)
        print 'Error is: ', err
        iter += 1

    return Lk, Sk

L, S = robust_pca(imgresh)

print L, S
print L.shape, S.shape

img_out = np.zeros(h*w*3)
img_out2 = np.zeros(h*w*3)
# generate output
for index in range(num):
    img_out += L[:, index]
    img_out2 += S[:, index]

img_out = np.reshape((img_out/num), (h, w, 3))
img_out2 = np.reshape(img_out2, (h, w, 3))
err = la.norm(img_out.astype("float") - noreflection.astype("float"))
print err #
cv2.imwrite('background_pca_frpca.png', img_out)
cv2.imwrite('reflection_pca_frpca.png', img_out2)
