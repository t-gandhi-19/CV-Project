import numpy as np

def sum_abs_dist(src = None, dst = None):
    '''computes sum of absoulte difference error between two patches
    input src - M,KxK,3
    dst - N,KxK,3'''
    num_dim_req = [3,3]
    if src.ndim != num_dim_req[0] or dst.ndim != num_dim_req[1]:
        raise ValueError("src and dst must have 3 dimensions")

    k_k = [dst.shape[1:]]
    if src.shape[1:] != k_k[0]:
        raise ValueError("src and dst must have the same shape")
    
    #add axis to the arrays to allow bradcasting
    new_array_src = np.zeros((src.shape[0], 1, src.shape[1], src.shape[2]))
    new_array_dst = np.zeros((1, dst.shape[0], dst.shape[1], dst.shape[2]))

    print("first for loop")
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for k in range(src.shape[2]):
                new_array_src[i][0][j][k] = src[i][j][k]

    print("second for loop")
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            for k in range(dst.shape[2]):
                new_array_dst[0][i][j][k] = dst[i][j][k]
                
    if new_array_src.shape != (src.shape[0], 1, src.shape[1], src.shape[2]):
        raise ValueError("src must have shape M,1,KxK,3")
    if new_array_dst.shape != (1, dst.shape[0], dst.shape[1], dst.shape[2]):
        raise ValueError("src must have shape 1,N,KxK,3")

    src = new_array_src
    dst = new_array_dst
    
    #compute sum of absolute difference

    sum_abs_d = np.zeros((src.shape[0], dst.shape[1]))
    sad = np.zeros((src.shape[0], dst.shape[1]))
    print("third for loop")
    for channel in range(3):
        print("channel: ", channel)
        # sum_abs_d += np.sum(np.abs(src[:,:,:,channel] - dst[:,:,:,channel]), axis = 2)
        # sum_abs_d = sum_abs_d + 
    
    return sum_abs_d


# a = np.random.rand(4,6,3)
# b =  np.random.rand(5,6,3)
# print(a)
# print(b)
# print(sum_abs_dist(a,b)[0])
# print(sum_abs_dist(a,b)[1])