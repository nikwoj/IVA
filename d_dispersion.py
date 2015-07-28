disper = np.zeros((K, K, N))
for n in range(N) :
    for k in range(K) :
        for k2 in range(k) :
            ## Indexing is a terrible thing...
            value = np.dot(X[n,:,k], X[n,:,(k-k2)].T) / N
            disper[N*k : (N+1)*k , N*(k-k2) : (N+1)*(k-k2), n] = value
            disper[N*(k-k2) : (N+1)*(k-k2) , N*k : (N+1)*k, n] = value.T
            