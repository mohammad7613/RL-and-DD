import numpy as np

def channels_specs(file_loc):
    
    file = open(file_loc, 'r')

    raw_file = file.read()
    channel_name = raw_file.replace('\t', ' ').split("\n")

    file.close()
    
    return channel_name

def OrderEstimate_byChannels(Data, channels, max_order, min_order, leap_length):

    # This channel estimate the order of AR process between channels,
    # Data -> m-by-n matrix, m is number of channels and n is length of data,
    # channels -> channels of Data matrix which orders must be calculated (list with maximum length m),
    # max_order -> maximum valid order to be included in estimation (less than n)
    # min_order -> minimum valid order to be included in estimation (more than zero)
    # leap_length -> orders likelihood will be computed with this leap_length
    
    number_of_channels = len(channels)

    ctr = 0
    
    orders_mat = np.zeros((number_of_channels, number_of_channels))

    orders = np.arange(min_order, max_order, leap_length)
    
    for a, channel_a in enumerate(channels):

        for b, channel_b in enumerate(channels):

            if a != b:

                BICs = []

                x_t = Data[channel_a, :]
                y_t = Data[channel_b, :]

                for order in orders:

                    a_est_mul, b_est_mul = mulvar_AR_est(x_t, y_t, order, order)

                    x_t_rec_mat = x_t_recun_ab(a_est_mul, b_est_mul, x_t, y_t)
                    BICs.append(BIC_calc(x_t, x_t_rec_mat, order))
                    
                # plt.plot(BICs)
                # plt.show()
                
                ctr = ctr + 1
                print(int(ctr / (number_of_channels * (number_of_channels - 1)) * 100), "%", "Estimated Order is", orders[int(np.argmin(BICs))])

                orders_mat[a, b] = orders[int(np.argmin(BICs))]
                
    return orders_mat

def a_estimation_err(a_est, x_t):
    
    order = len(a_est)
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:order] = x_t[:order]

    for i in range(order, length):

        for j in range(order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)

def x_t_recun(a_est, x_t):
    
    order = len(a_est)
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:order] = x_t[:order]

    for i in range(order, length):

        for j in range(order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]

    return x_t_rec

def ab_estimation_err(a_est, b_est, x_t, y_t):
    
    a_order = len(a_est)
    b_order = len(b_est)
    
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:a_order] = x_t[:a_order]

    for i in range(a_order, length):

        for j in range(a_order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]
            
        for j in range(b_order):

            x_t_rec[i] = x_t_rec[i] + b_est[j] * y_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)

def x_t_recun_ab(a_est, b_est, x_t, y_t):
    
    a_order = len(a_est)
    b_order = len(b_est)
    
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:a_order] = x_t[:a_order]

    for i in range(a_order, length):

        for j in range(a_order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]
            
        for j in range(b_order):

            x_t_rec[i] = x_t_rec[i] + b_est[j] * y_t[i - j - 1]

    return x_t_rec


def univar_AR_est(x_t, order):
    
    length = len(x_t)

    X_mat = np.zeros((length - order, order))
    X_vec = np.zeros((length - order))

    for i in range(length - order):

        X_mat[i, :] = x_t[i : i + order]
        X_vec[i] = x_t[i + order]

    return np.flip(np.linalg.pinv(X_mat) @ X_vec)

def mulvar_AR_est(x_t, y_t, a_order, b_order):
    
    length = len(x_t)

    X_mat = np.zeros((length - a_order, a_order + b_order))
    X_vec = np.zeros((length - a_order))

    for i in range(length - a_order):

        X_mat[i, : a_order] = x_t[i : i + a_order]
        X_mat[i, a_order : a_order + b_order] = y_t[i : i + b_order]
        
        X_vec[i] = x_t[i + a_order]

    coef_est = np.linalg.pinv(X_mat) @ X_vec
    
    a_est = np.flip(coef_est[:a_order])
    b_est = np.flip(coef_est[a_order : a_order + b_order])
    
    return a_est, b_est

def GC_calc(x_t, y_t, order):

    # it is much better to give access to orders to users!
    GC_val = []
    univar_error = []
    mulvar_error = []

    a_est_uni = univar_AR_est(x_t, order)

    a_order = order
    b_order = a_order

    a_est_mul, b_est_mul = mulvar_AR_est(x_t, y_t, a_order, b_order)

    univar_error.append(a_estimation_err(a_est_uni, x_t))
    mulvar_error.append(ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t))
        
    GC_val.append(np.log(univar_error[-1] / mulvar_error[-1]))
        
    # print("Order is", order, "and Granger Causality is", np.log(a_estimation_err(a_est_uni, x_t) / ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t)), "Univar Error is", a_estimation_err(a_est_uni, x_t), "and mulvar error is", ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t))
        
    return GC_val, univar_error, mulvar_error

def AIC_calc(y, y_pred, k):
    
    # AIC = 2k + n * ln(mean sum of residuals)
    
    n = len(y)
    
    if len(y) != len(y_pred):
        
        print("Predicted values and real data doesn't have same length")
        
        return ''
    
    MSR = np.sum((y - y_pred) ** 2) / n
    
    return 2 * k + n * np.log(MSR)

def BIC_calc(y, y_pred, k):
    
    # BIC = k * ln(n) + n * ln(mean sum of residuals)
    
    n = len(y)
    
    if len(y) != len(y_pred):
        
        print("Predicted values and real data doesn't have same length")
        
        return ''
    
    MSR = np.sum((y - y_pred) ** 2) / n
    
    return k * np.log(n) + n * np.log(MSR)

def GrangerCausalityEstimator(Data, channels, window_length, overlap_ratio, orders_mat):

    # # This Function calculates the Granger Causality between different channels of Data
    # Data -> m-by-n data matrix, m is number of channels and n is length of data
    # channels -> list of channels to calculate GC
    # window_length -> Length of windows to compute GC over them
    # overlap_ratio -> Precent of overlap between consequent windows
    # orders_mat -> order of AR process estimation between each two channel (must be a m-by-m channel)
    # it is considered to be output of function 'OrderEstimate_byChannels' above
    
    number_of_channels, N = Data.shape
    number_of_windows = int((N - window_length) / ((1 - overlap_ratio) * window_length)) + 1

    GC_values = np.zeros((number_of_windows, number_of_channels, number_of_channels))

    for win_step in range(number_of_windows):

        print("In Progress", win_step / number_of_windows * 100, "% ...")

        for i, channel_a in enumerate(channels):

            for j, channel_b in enumerate(channels):

                if i != j:

                    win_stp = int((win_step) * (1 - overlap_ratio) * window_length)
                    win_enp = win_stp + window_length

                    x_t = Data[channel_a, win_stp : win_enp]
                    y_t = Data[channel_b, win_stp : win_enp]

                    est_order = int(orders_mat[i, j])

                    tmp, tmp_err1, tmp_err2 = GC_calc(x_t, y_t, est_order)

                    GC_values[win_step, i, j] = tmp[0]
                    
    return GC_values

def a_estimation_RW(x_t, order, var, max_iter):
    
    # order = 4
    a_est = np.random.normal(0, 2, size = (order))
    # var = 1
    do_it = 1
    # max_iter = 200
    iteration_i = 0
    iteration_j = 0

    while do_it:

        est_err = a_estimation_err(a_est, x_t)
        iterate = 1

        while iterate:

            a_est_ = a_est + np.random.normal(0, var / np.log(iteration_i + 2), size = (order))

            if a_estimation_err(a_est_, x_t) < est_err:

                a_est = a_est_
                iterate = 0

            elif iteration_j > max_iter * 50:

                iterate = 0
                do_it = 0

            iteration_j = iteration_j + 1

        if iteration_i > max_iter:

            do_it = 0

        iteration_i = iteration_i + 1

        print(iteration_i, a_est, est_err)
    
    return a_est

def ab_estimation_RW(x_t, y_t, a_order, b_order, var, max_iter):
    
    a_est = np.random.normal(0, 2, size = (a_order))
    b_est = np.random.normal(0, 2, size = (b_order))
    
    do_it = 1
    
    iteration_i = 0
    iteration_j = 0

    while do_it:

        est_err = ab_estimation_err(a_est, b_est, x_t, y_t)
        iterate = 1

        while iterate:

            a_est_ = a_est + np.random.normal(0, var / np.log(iteration_i + 2), size = (a_order))
            b_est_ = b_est + np.random.normal(0, var / np.log(iteration_i + 2), size = (b_order))

            if ab_estimation_err(a_est_, b_est_, x_t, y_t) < est_err:

                a_est = a_est_
                b_est = b_est_
                iterate = 0

            elif iteration_j > max_iter * 50:

                iterate = 0
                do_it = 0

            iteration_j = iteration_j + 1

        if iteration_i > max_iter:

            do_it = 0

        iteration_i = iteration_i + 1

        print(iteration_i, a_est, b_est, est_err)
    
    return a_est, b_est


def AR_Generator(order, a_order, b_order, length):

    # This function generates a random Auto-Regressive signal
    
    invalid_output = 1

    while invalid_output:

        x_t4 = np.zeros(length)
        x_t = np.zeros(length)

        x_t4[:a_order] = np.random.normal(0, 5, size = (a_order))

        x_t[:order] = np.random.normal(0, 5, size = (order))

        a = np.random.normal(0, 0.5, size = (order))

        a_4 = np.random.normal(0, 0.5, size = (a_order))
        b = np.random.normal(0, 0.5, size = (b_order))

        for i in range(order, length):

            for n, a_i in enumerate(a):

                x_t[i] = x_t[i] + a_i * x_t[i - n - 1]


        for i in range(a_order, length):

            for n, a_4i in enumerate(a_4):

                x_t4[i] = x_t4[i] + a_4i * x_t4[i - n - 1]

            for n, b_i in enumerate(b):

                x_t4[i] = x_t4[i] + b_i * x_t[i - n - 1]

        if np.abs(x_t4[-1]) + np.abs(x_t[-1]) < 1000000:

            invalid_output = 0
            
    return x_t4, x_t

def AR_Process_Gen(init_vals, coeffs, N):
    
    # This Function Generate an Auto-Regressive process based on initial values and coefficients determined in the inputs
    # length of the signal is N, Consider this point that stability of signal depends on these inputs.
    
    if len(init_vals) != len(coeffs):
        
        print("INVALID input, order not detemined!")
        return 0
    
    order = len(init_vals)
    
    AR_signal = np.zeros(N)
    AR_signal[:order] = init_vals
    
    for i in range(order, N):
        
        for n in range(order):
            
            AR_signal[i] = AR_signal[i] + coeffs[n] * AR_signal[i - n - 1]
            
    return AR_signal

def AR_Cross_Process_Gen(init_vals, coeffs, y, orders_vec):
    
    # This Function Generate an Auto-Regressive process based on initial values and coefficients determined in the inputs
    # length of the signal is N, Consider this point that stability of signal depends on these inputs.
    
    # This function calculates AR_Cross_Process using other signals and also init_values of it self
    
    number_of_crosses, N = y.shape
    
    if len(init_vals) != len(coeffs):
        
        print("INVALID input, order not detemined!")
        return 0
    
    order = len(init_vals)
    
    AR_signal = np.zeros(N)
    AR_signal[:order] = init_vals
    
    for i in range(order, N):
        
        for n in range(order):
            
            AR_signal[i] = AR_signal[i] + coeffs[n] * AR_signal[i - n - 1]
            
        for n_cross in range(number_of_crosses):
            
            for n in range(len(orders_vec[n_cross, :])):
                
                AR_signal[i] = AR_signal[i] + orders_vec[n_cross, n] * y[n_cross, i - n - 1]
            
    return AR_signal

def movmean(A, k):

    win_length = k
    MAFA = []
    
    if len(A) < k:

        print("I don't know how to calculate this, The window must be less than length. So the length forced to be len(A)")

        win_length = len(A)

    new_elements_len = len(A) - win_length + 1

    for element in range(new_elements_len):

        MAFA.append(np.mean(A[element : (element + 1) + k ]))

    return np.array(MAFA)

def Accuracy_Performance(X):

    Acc = []

    for i in range(len(X)):

        Acc.append(1 - np.sum(X[:i]) / (i + 1))

    return Acc

