import numpy as np

def main():
    # Horizontal vectors
    # creating vector 1
    v1 = np.array([1,1,1])
    # creating vector 2
    v2 = np.array([1,2,3])

    print("Printing V1: ",v1,"Printing V2: ",v2)

    # vertical vectors
    # Using reshape attribute to change the dimension of the array
    vector1 = v1.reshape(3,1)
    vector2 = v2.reshape(3,1)
    print("vertical vectors")
    print("Printing v1: \n",vector1 , "\n Printing v2: \n",vector2)

    # creating a matriz
    row1 = [-2, -4, 2]
    row2 = [-2, 1, 2]
    row3 = [4, 2, 5]
    matriz = np.array([row1,row2,row3])
    print ("Matriz = \n", matriz)


if __name__ == '__main__':
    main()