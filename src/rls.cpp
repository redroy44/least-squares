#include "ls.h"
#include <iostream>
#include <armadillo>

namespace ls {
  RecursiveLeastSquares::RecursiveLeastSquares() {
    std::cout << "Hello from RecursiveLeastSquares library" <<std::endl;
    matrix = arma::randu<arma::mat>(3,3);
  }

  RecursiveLeastSquares::~RecursiveLeastSquares() {
	// TODO Auto-generated destructor stub
  }
  void RecursiveLeastSquares::printMat() const {
    std::cout << matrix << std::endl;
  }
}
