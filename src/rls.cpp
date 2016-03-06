#include "ls.h"
#include <iostream>
#include <armadillo>

namespace ls {
   RecursiveLeastSquares::RecursiveLeastSquares(std::string s) {
      std::cout << s << std::endl;
   }

  RecursiveLeastSquares::RecursiveLeastSquares(const unsigned int &order) : RecursiveLeastSquares("RLS online estimation") {
    m_order = order;
    m_inv_regression_matrix = 1000 * arma::eye(order, order);
    m_phi = arma::zeros(order);
    m_theta = arma::zeros(order);
    m_aux_vector = arma::zeros(order);
    m_driving_noise = 0;
    m_signal_length = 0;
    m_error = 0;
  }

  void RecursiveLeastSquares::print() const {
    IEstimator::print();
    std::cout << "Model order: " << m_order << std::endl;
    std::cout << "Inverse Regression matrix: \n" << m_inv_regression_matrix << std::endl;
    std::cout << "Inverse Regression matrix determinant: " << arma::det(m_inv_regression_matrix) << std::endl;
    std::cout << "Auxiliary vector: \n" << m_aux_vector << std::endl;
  }

  void RecursiveLeastSquares::estimate() {
     IEstimator::print();
     std::cout << m_order << std::endl;
  }

  void RecursiveLeastSquares::estimate(const arma::vec &phi, const double &sample) {
    m_phi = phi;
    m_error = sample - as_scalar(m_phi.t() * m_theta);
    m_aux_vector = (m_inv_regression_matrix * m_phi)/repmat((1 + m_phi.t()*m_inv_regression_matrix*m_phi),m_order, 1);
    m_theta = m_theta + m_aux_vector * m_error;
    m_inv_regression_matrix = m_inv_regression_matrix - (m_inv_regression_matrix * m_phi * m_phi.t() * m_inv_regression_matrix)/repmat((1 + m_phi.t() * m_inv_regression_matrix * m_phi), m_order, m_order);

  m_driving_noise = 0.01;
  }

}
