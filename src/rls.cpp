#include "ls.h"
#include <iostream>
#include <armadillo>

namespace ls {
  RecursiveLeastSquares::RecursiveLeastSquares(const unsigned int &order) {
    std::cout << "Hello from RecursiveLeastSquares library" <<std::endl;
  }

  void RecursiveLeastSquares::print() const {
     IEstimator::print();
     std::cout << m_order << std::endl;
  }

  void RecursiveLeastSquares::estimate() {
     IEstimator::print();
     std::cout << m_order << std::endl;
  }

  void RecursiveLeastSquares::estimate(const arma::vec &phi, const double &sample) {
     IEstimator::print();
     std::cout << m_order << std::endl;
  }
}
