import numpy as np
import numpy.linalg as l

def exercise0():
    # Horizontal vectors
    # creating vector 1
    # array makes arrays of 1-D or n-D
    print("\n*** exercise 0 ***")
    v1 = np.array([1,1,1])
    # creating vector 2
    v2 = np.array([1,2,3])

    print("\nPrinting V1: ",v1,"Printing V2: ",v2)

    # vertical vectors
    # Using reshape attribute to change the dimension of the array (row,column)
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

    print("\n*** exercise 1 ***")

    #Using norm method from Linear Algebra module
    #axis=1 will allow us to get the norm by rows from the matrix
    #which means we are taking the norm from each vector of the matrix

    # l-infinity
    matrix_infinity_norm = l.norm(matrix, np.inf, axis=1)
    print("norm l-infinity per vector: ", matrix_infinity_norm)
    #norm l0
    matrix_norml0 = l.norm(matrix, 0, axis=1)
    print("norm l0 per vector: ", matrix_norml0)
    #norm l1
    matrix_norml1 = l.norm(matrix, 1, axis=1)
    print("norm l1 per vector: ", matrix_norml1)
    #norm l2
    matrix_norml2 = l.norm(matrix, 2, axis=1)
    print("norm l2 per vector: ", matrix_norml2)
    
    
    return matrix_norml2

def exercise2(norms, matrix):

    print("\n*** exercise 2 ***")

    # sorting norm l2
    array_norms = np.array(norms)
    array_norms[::-1].sort() #mutating the array for decreasing. instead of creating a new array
    print("sorting numbers: ",array_norms)

    #checking the size of the array
    size = array_norms.size
    print("size of the array: ", size)

    #Reordering the new matrix
    new_matrix = []
    values = {}
    i = 0
    for i in range(size):
        val = str("x"+str(i)) 
        values[val] = array_norms[i]

        if values[val] == l.norm(matrix[i], 2):
            new_matrix.append(matrix[i])
        elif values[val] == l.norm(matrix[i-1], 2):
            new_matrix.append(matrix[i-1])
        else:
            new_matrix.append(matrix[i-2])
    
    # creating the original matrix
    new_matrix = np.array(new_matrix)
    print("original matrix: \n",matrix)
    print("matrix with new order from norm l2: \n",new_matrix)

    return

def exercise3(matrix):
    
    print("\n*** exercise 3 ***")

    # matrix
    print("\nMatrix A: \n",matrix)
    
    # taking the media from matrix A
    media = np.mean(matrix, axis=0)
    print("\nthe media of the matrix A is: ",media)

    # taking the desviacion standard from matrix A
    dst = np.std(matrix, axis=0)
    print("\nthe desviacion standard of the matrix A is: ",dst)

    # rest each mean value to it's correspond column on the matrix
    rest_array = matrix - media
    print("\nthe rest of the each media value for matriz colums is:  \n",rest_array)

    # divide each desviacion standard value to it's correspond column on the matrix
    div_array = matrix/dst
    print("\nthe division of the each desviacion standard value for matriz colums is:  \n",div_array)

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
        print("matrix no singular")
    else:
        print("nmatrix is singular")
    
    # calculating the traze of the matrix
    matrix_trace = np.trace(matrix)
    print("\ncalculating the traze of the matrix: ",matrix_trace)

     # calculating the eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = l.eig(matrix)
    print("\neigenvalues: ",eigenvalues)
    print("\neigenvector: \n",eigenvectors)

    return

def exercise5():
    
    print("\n*** exercise 5 ***")

    #vectors
    x1 = np.array([1,2,3])
    x2 = np.array([4,5,6])
    x3 = np.array([7,8,9])
    X = np.array([x1,x2,x3])

    print("X Vectors: x1: ",x1,"x2: ",x2,"x3: ",x3)

    c1 = np.array([1,0,0])
    c2 = np.array([0,1,1])
    C = np.array([c1,c2])

    print("C Vectors: c1: ",c1,"c2: ",c2)

    #calculating the distance between the X vectors and C vectors.
    dist01 = l.norm(x1 - c1)
    print ("distance between x1 and c1: ", dist01)
    dist02 = l.norm(x1 - c2)
    print ("distance between x1 and c2: ", dist02)
    dist03 = l.norm(x2 - c1)
    print ("distance between x2 and c1: ", dist03)
    dist04 = l.norm(x2 - c2)
    print ("distance between x1 and c2: ", dist04)
    dist05 = l.norm(x3 - c1)
    print ("distance between x1 and c1: ", dist05)
    dist06 = l.norm(x3 - c2)
    print ("distance between x2 and c2: ", dist06)


def main():
    #exercise0
    matrix = exercise0()
    print ("\nmatrix = \n", matrix)

    #exercise1
    norms = exercise1(matrix)

    #exercise2
    exercise2(norms, matrix)

    #exercise3
    exercise3(matrix)

    #exercise4
    exercise4(matrix)

    #exercise5
    exercise5()
    

if __name__ == '__main__':
    main()