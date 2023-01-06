import numpy as np
import numpy.linalg as l

def exercise0():
    # Horizontal vectors
    # creating vector 1
    print("\n*** exercise 0 ***")
    v1 = np.array([1,1,1])
    # creating vector 2
    v2 = np.array([1,2,3])

    print("\nPrinting V1: ",v1,"Printing V2: ",v2)

    # vertical vectors
    # Using reshape attribute to change the dimension of the array
    vector1 = v1.reshape(3,1)
    vector2 = v2.reshape(3,1)
    print("vertical vectors")
    print("Printing v1: \n",vector1 , "\n Printing v2: \n",vector2)

    # creating a matrix
    row1 = [-2, -4, 2]
    row2 = [-2, 1, 2]
    row3 = [4, 2, 5]
    matrix = np.array([row1,row2,row3])
    
    return matrix 


def exercise1(matrix):
    # Using the Linear Algebra module for numpy linalg
    # taking vectors from the matrix
    print("\n*** exercise 1 ***")
    v1 = matrix[0, :]
    v2 = matrix[1, :]
    v3 = matrix[2, :]
    print("\nVectors from the previous matrix \n")
    print("V1: ",v1,"V2: ",v2,"V3: ",v3)

    # l-infinity
    infinity_norm1 = l.norm(v1, np.inf)
    infinity_norm2 = l.norm(v2, np.inf)
    infinity_norm3 = l.norm(v3, np.inf)
    print("\n norm l-infinity for v1: ", infinity_norm1,"for v2: ", infinity_norm2,"for v3: ", infinity_norm3)
    
    #norm l0
    norml0_v1 = l.norm(v1, 0)
    norml0_v2 = l.norm(v2, 0)
    norml0_v3 = l.norm(v3, 0)
    print("\n norm l0 for v1: ", norml0_v1,"for v2: ", norml0_v2,"for v3: ", norml0_v3)

    #norm l1
    norml1_v1 = l.norm(v1, 1)
    norml1_v2 = l.norm(v2, 1)
    norml1_v3 = l.norm(v3, 1)
    print("\n norm l1 for v1: ", norml1_v1,"for v2: ", norml1_v2,"for v3: ", norml1_v3)

    #norm l2
    norml2_v1 = l.norm(v1, 2)
    norml2_v2 = l.norm(v2, 2)
    norml2_v3 = l.norm(v3, 2)
    print("\n norm l2 for v1: ", norml2_v1,"for v2: ", norml2_v2,"for v3: ", norml2_v3)
    
    return norml2_v1, norml2_v2, norml2_v3

def exercise2(norms):

    print("\n*** exercise 2 ***")

    # sorting norm l2
    array_norms = np.array(norms)
    array_norms[::-1].sort()
    print("\n sorting numbers: ",array_norms)
    
    #TO FINISH
    # creating the original matrix
    v1 = [-2, -4, 2]
    v2 = [-2, 1, 2]
    v3 = [4, 2, 5]
    matrix = np.array([v3,v1,v2])
    print("\n matrix from norm l2: \n",matrix)

    return

def exercise3(matrix):
    
    print("\n*** exercise 3 ***")
    
    # taking the media from matrix A
    media = np.mean(matrix, axis=0)
    print("\n the media of the matrix A is: ",media)

    # taking the desviacion standard from matrix A
    dst = np.std(matrix, axis=0)
    print("\n the desviacion standard of the matrix A is: ",dst)

    return

def exercise4(matrix):
    
    print("\n*** exercise 4 ***")
    
    # Matrix inverse
    matrix_inv = l.inv(matrix)
    print("\nMatrix inverse: \n",matrix_inv)

    # verification
    # If multiply the original matrix against the inverse we should get the identity
    matrix_mult = np.matmul(matrix,matrix_inv)
    print("\nIf multiply the original matrix against the inverse we should get the identity: \n",matrix_mult)

    # Calculating the determinante
    matrix_det = l.det(matrix)
    print("\nCalculating the determinante: ",matrix_det)
    
    # verification
    if matrix_det != 0:
        print("matrix no singular\n")
    else:
        print("nmatrix is singular\n")
    
    # calculating the traze of the matrix
    matrix_trace = np.trace(matrix)
    print("\ncalculating the traze of the matrix: ",matrix_trace)

     # calculating the eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = l.eig(matrix)
    print("\neigenvalues: ",eigenvalues)
    print("\neigenvector: \n",eigenvectors)

def exercise5():
    pass

def main():
    #exercise0
    matrix = exercise0()
    print ("\nmatrix = \n", matrix)

    #exercise1
    norms = exercise1(matrix)

    #exercise2
    exercise2(norms)

    #exercise3
    exercise3(matrix)

    #exercise4
    exercise4(matrix)

    #exercise5
    exercise5()
    

if __name__ == '__main__':
    main()