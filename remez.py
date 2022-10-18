"""
Remez Algorithm - znajdowanie n-tego wielomianu optymalnego w sensie aproksymacji jednostajnej dla funkcji ciągłej określonej na odcinku [a,b].
Autor: Maciej Szczutko
Program jest czescia pracy licencjackiej pt. "Aproksymacja jednostajna i metoda Remeza" napisanej pod opieką dr. hab. Pawła Woźnego
na Uniwersytecie Wrocławskim.
Wejscie:
a,b - lewy i prawy koniec przedzialu
n - stopien wielomianu optymalnego
f(x) wzor jawny funkcji f (ograniczony do funkcji z modulu math)
Wyjscie:
Wspolczynniki w bazie potegowej, alternans, wartosci funkcji bledu w punktach alternansu
Wykres1: funkcja f, wielomian optymalny
Wykres2: Funkcja bledu
"""


import numpy as np
from scipy import linalg, optimize
from math import *
from root_finder import *
import matplotlib.pyplot as plt


def f(x):  # function to aprroximate
    return sin(x**3+1/x)


#interval
a = 0.5
b = 2
n = 10  #degree of approximation
max_iter = 100
copiable = False  #set to True to easy copy in standard form


def make_matrix(array):  # 1-D numpy array
    matrix = np.vstack((array**0, array))
    for i in range(n - 1):
        matrix = np.vstack((matrix, matrix[-1, ] * array))
    matrix = np.hstack(
        (matrix.T, np.matrix([(-1)**(k % 2) for k in range(n + 2)]).T))
    return (matrix)


def horner(x):
    result = 0
    for coefficient in reversed(coef_L):
        result = result * x + coefficient
    return result


def r(x):
    return f(x) - horner(x)


def r_(x):
    return -r(x)


R = np.vectorize(r)
R_ = np.vectorize(r_)
F = np.vectorize(f)  # możemy teraz podawac cala tablice argumentow do funkcji (tylko dla wygody)
H = np.vectorize(horner)


def print_poly(L):
    string = ""
    for n, coefficient in enumerate(L):
        if coefficient > 0:
            string += "+"
        string += "{}*x^{}".format(format(coefficient, '.32f'), n)
    print(string)
X = np.linspace(a, b, n + 2)  #initalize train set
iter = 0
M, m = 2, 1
while M > (1 + 5 * 10e-6) * m and iter < max_iter:
    A = make_matrix(X)  # initialize matrix
    B = F(X)  #right hand side vector
    solution = linalg.solve(A, B)
    coef_L, d = solution[:-1], solution[-1]
    Z = np.array([a])
    for i in range(len(X) - 1):
        roots, *kwargs = find_all_roots(R, X[i], X[i + 1])
        Z = np.append(Z, roots[0])
    Z = np.append(Z, b)
    Y = np.array([])  # new_control points
    for i in range(len(Z) - 1):
        if r(X[i]) > 0:
            m = optimize.minimize_scalar(R_,
                                         bounds=(Z[i], Z[i + 1]),
                                         method='bounded')
            if (-m.fun > R(Z[i]) and -m.fun > R(Z[i + 1])):
                y = m.x
            elif R(Z[i]) > R(Z[i + 1]):
                y = Z[i]
            else:
                y = Z[i + 1]
        else:
            m = optimize.minimize_scalar(R,
                                         bounds=(Z[i], Z[i + 1]),
                                         method='bounded')
            if (m.fun < R(Z[i]) and m.fun < R(Z[i + 1])):
                y = m.x
            elif R(Z[i]) < R(Z[i + 1]):
                y = Z[i]
            else:
                y = Z[i + 1]
        Y = np.append(Y, y)
    eval = abs(R(Y))
    m, M = min(eval), max(eval)
    ratio = abs(M / m)
    X = np.array(Y)
    iter += 1
print("Bład aproksymacji wynosi {}".format(abs(d)))
print("Wykonanych iteracji: {}".format(iter))
print("Współczynniki {}-ego wielomianu optymalnego: {}".format(n, coef_L))
print("Punkty alternansu: {}".format(Y))
print("Wartosci w punktach alternansu: {}".format(R(Y)))
if copiable:
    print_poly(coef_L)

#Ploting result
k = 10**4
u = np.linspace(a, b, k)
y_f = F(u)
y_P = H(u)
plt.plot(u, y_f, label="$f(x)$")
plt.plot(u, y_P, label="$P^*_{" + str(n) + "}(x)$")
plt.legend()
plt.show()
plt.plot(u, y_f - y_P)
plt.plot([a, b], [d, d], color="red")
plt.plot([a, b], [-d, -d], color="red")
plt.title('Wykres funkcji błędu')
plt.show()
