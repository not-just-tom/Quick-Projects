import numpy as np
import pprint
import scipy.linalg
import scipy

# Gaussian elimination requires 2/3n^3 floating point operations 

A = np.array([ [7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6] ])
P, L, U = scipy.linalg.lu(A)

print("A:")
pprint.pprint(A)

print("P:")
pprint.pprint(P)

print("L:")
pprint.pprint(L)

print("U:")
pprint.pprint(U)

# under the hood


def mult_matrix(A, B):
     
    result = [[0.0] * clen(B[0.0]) for i in range(len(A))]
    
    for i in range(len(A)):
        
        # iterating by column by B
        for j in range(len(B[0.0])):
            
            
            # iterating by rows of B
            for k in range(len(B)):
                
                result[i][j] += float(A[i][k]) * float(B[k][j])
    return result    

def pivot_matrix(P):
    m = len(P)

    id_mat = [[float(i ==j) for i in range(m)] for j in range(m)]

    for j in range(m):
        row = max(range(j,m), key = lambda i: (P[i][j]))
        if j != row:
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

    return id_mat

def LU_Doolittle(A):
    n = len(A)

    U = [[0.0] * n for i in range(n)] 
    L = [[0.0] * n for i in range(n)]
    P = pivot_matrix(A)
    PA = mult_matrix(P, A)
    for j in range(n):
        L[j][j] = 1.0

        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1
            
                                                                                                                                                               
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]
            
    return (P, L, U)
    
P2, L2, U2 = LU_Doolittle(A)

print("And now for something exactly the same")
print("P:")
pprint.pprint(P2)
print("L:")
pprint.pprint(L2)
print("U:")
pprint.pprint(U2)