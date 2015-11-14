#include <iostream>
#include "ls.h"
#include <armadillo>

int main () {
ls::LeastSquares obj = ls::LeastSquares();
ls::RecursiveLeastSquares obj2 = ls::RecursiveLeastSquares();

obj2.printMat();

return 0;
}
