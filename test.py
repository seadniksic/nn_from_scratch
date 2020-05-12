import numpy as np

test = np.array([[1,2,3],[4,5,6]])
test4 = np.array([[1], [2], [3]])

print(np.dot(test, test4))

#test = np.insert(test, 2, 3, axis = 1) 

test2 = np.array([1,2,3])
gib = np.array([4,5,6])
test3 = np.array([])
print(test3.size)


test8 = test.flatten()

print(test8)
print(np.concatenate((test8, test2)))