cimport cython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef extract(double[:,:] xs, double[:,:] ys, double[:,:] xs_008, double[:,:] ys_008):
    print("xs_008", np.shape(xs_008))
    print("ys_008", np.shape(ys_008))
    t2 = time.time()
    batch = 0
    for i in range(len(xs[batch])):
        if i == 0:
            # print("xs_008 before arrange:", xs * 8)
            x_008 = torch.arange(int(xs_008[batch][i]+0.5)-24, int(xs_008[batch][i]+0.5)+24)
            # print("xs_008 after arrange:", x_008)
            # x_004 = torch.range(int(xs_004[batch][i]+0.5)-24, int(xs_004[batch][i]+0.5)+24)
        else:
            x_buff_008 = torch.arange(int(xs_008[batch][i]+0.5)-24, int(xs_008[batch][i]+0.5)+24)
            # print("xs_008 size:", np.shape(xs_008))
            # x_buff_004 = torch.range(int(xs_004[batch][i]+0.5)-24, int(xs_004[batch][i]+0.5)+24)
            for alpha in x_buff_008:
                alpha = torch.tensor([alpha])
                if alpha not in x_008:
                    x_008 = torch.cat((x_008, alpha))
                else:
                    continue
        #t3 = time.time()
        #print("t3", t3 - t11)
        for j in range(len(ys[batch])):
            if j == 0:
                y_008 = torch.arange(int(ys_008[batch][i]+0.5)-24, int(ys_008[batch][i]+0.5)+24)
            else:
                y_buff_008 = torch.arange(int(ys_008[batch][i]+0.5)-24, int(ys_008[batch][i]+0.5)+24)
                for beta in y_buff_008:
                    beta = torch.tensor([beta])
                    if beta not in y_008:
                        y_008 = torch.cat((y_008, beta))
                    else:
                        continue

        print(np.shape(y_008))