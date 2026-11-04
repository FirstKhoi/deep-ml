import numpy as np

def posterior(X, y, m0, S0, sigma2):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m0 = np.array(m0, dtype=float)
    S0 = np.array(S0, dtype=float)
    sigma2 = float(sigma2)
    
    # 1. Tính nghịch đảo của hiệp phương sai tiên nghiệm S0
    inv_S0 = np.linalg.inv(S0)

    # 2. inv_SN = inv_S0 + (1 / sigma2) * X^T * X
    inv_SN = inv_S0 + (1.0 / sigma2) * (X.T @ X)
    SN = np.linalg.inv(inv_SN)

    # 3. mN = SN * (inv_S0 * m0 + (1 / sigma2) * X^T * y)
    prio_term = inv_S0 @ m0
    data_term = (1.0 / sigma2) * (X.T @ y)
    mN = SN @ (prio_term + data_term)
    
    return mN, SN


def sequential_posterior(X, y, m0, S0, sigma2):
    # return (mN, SN, means)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m0 = np.array(m0, dtype=float)
    S0 = np.array(S0, dtype=float)
    sigma2 = float(sigma2)

    m, S = m0.copy(), S0.copy()
    means = []
    N = X.shape[0]

    for i in range(N):
        x_i = X[i : i + 1]
        y_i = y[i : i + 1]

        m, S = posterior(x_i, y_i, m, S, sigma2)
        means.append(m.copy())

    return m, S, np.array(means)


def predictive(X_test, mN, SN, sigma2):
    # return (mean, var)
    X_test = np.array(X_test, dtype=float)
    mean = X_test @ mN

    # Tính phương sai từng điểm: sigma^2 + x_*^T * SN * x_*
    param_uncertainty = np.sum((X_test @ SN) * X_test, axis=1)
    var = float(sigma2) + param_uncertainty

    return mean, var