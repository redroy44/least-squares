#include "ls.h"
#include <iostream>

namespace ls {
  LeastSquares::LeastSquares() {
    std::cout << "Hello from LeastSquares library" <<std::endl;
  }

  LeastSquares::LeastSquares(const arma::vec &signal, const unsigned int &order) : LeastSquares() {
     std::cout << "Signal length: " << signal.n_rows <<std::endl;
     m_signal = signal;
     m_order = order;
     m_regression_matrix = arma::zeros(order, order);
     m_phi = arma::zeros(order);
     m_theta = arma::zeros(order);
     m_aux_vector = arma::vec(order);

     // build regression matrix
     m_signal = arma::join_cols(arma::zeros(1), m_signal);
     arma::vec indices = arma::linspace(0, m_signal.n_rows - 1, m_signal.n_rows);
     indices = arma::join_cols(arma::zeros(order), indices);
     //indices.print();

     for(unsigned int i = 0; i < m_signal.n_rows; i++) {
       m_phi = indices.subvec(i + 1, i + order);
       m_phi.transform([&](double val) {return (m_signal(val));});
       //m_phi.print();
       //std::cout << std::endl;
       m_regression_matrix += m_phi * m_phi.t();
       if(i < signal.n_rows) {
         m_aux_vector += signal(i) * m_phi;
       }
     }
  }

  LeastSquares::~LeastSquares() {
	// TODO Auto-generated destructor stub
  }

  void LeastSquares::estimate() {
    m_theta = arma::solve(m_regression_matrix, m_aux_vector);
  }

  void LeastSquares::print() const {
     std::cout << "Model order: " << m_order << std::endl;
     std::cout << "Regression matrix: \n" << m_regression_matrix << std::endl;
     std::cout << "Regression matrix determinant: " << arma::det(m_regression_matrix) << std::endl;
     std::cout << "Auxiliary vector: \n" << m_aux_vector << std::endl;
     std::cout << "Theta: \n" << m_theta << std::endl;
  }
}
