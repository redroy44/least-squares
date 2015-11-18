#include "ls.h"
#include <iostream>

namespace ls {
  LeastSquares::LeastSquares(std::string s) {
    std::cout << "Hello from LeastSquares library" <<std::endl << s << std::endl << std::endl;
  }

  LeastSquares::LeastSquares(const arma::vec &signal, const unsigned int &order) : LeastSquares("LS offline estimation") {
     std::cout << "Signal length: " << signal.n_rows <<std::endl;
     m_signal = signal;
     m_order = order;
     m_regression_matrix = arma::zeros(order, order);
     m_phi = arma::zeros(order);
     m_theta = arma::zeros(order);
     m_aux_vector = arma::zeros(order);
  }

  LeastSquares::LeastSquares(const unsigned int &order) : LeastSquares("LS online estimation") {
     m_order = order;
     m_regression_matrix = arma::zeros(order, order);
     m_phi = arma::zeros(order);
     m_theta = arma::zeros(order);
     m_aux_vector = arma::zeros(order);
  }

  LeastSquares::~LeastSquares() {
	// TODO Auto-generated destructor stub
  }

  const arma::vec& LeastSquares::getTheta() const {
     return m_theta;
  }

  void LeastSquares::estimate() {
     // build regression matrix
     m_signal = arma::join_cols(arma::zeros(1), m_signal);
     arma::vec indices = arma::linspace(0, m_signal.n_rows - 1, m_signal.n_rows);
     indices = arma::join_cols(arma::zeros(m_order), indices);
     //indices.print();

     for(unsigned int i = 0; i < m_signal.n_rows; i++) {
       m_phi = indices.subvec(i + 1, i + m_order);
       m_phi = flipud(m_phi);
       m_phi.transform([&](double val) {return (m_signal(val));});
       //m_phi.print();
       //std::cout << std::endl;
       m_regression_matrix += m_phi * m_phi.t();
       if(i < m_signal.n_rows - 1) {
         m_aux_vector += m_signal(i + 1) * m_phi;
       }
     }
    m_theta = arma::solve(m_regression_matrix, m_aux_vector);
  }

  void LeastSquares::estimate(const boost::circular_buffer<double> &phi, const double &sample) {
   fill_vector(phi);
   m_regression_matrix += m_phi * m_phi.t();
   if(arma::det(m_regression_matrix) < 1)
      std::cout << "det A: " << arma::det(m_regression_matrix) << std::endl;
   m_aux_vector += sample * m_phi;
   m_theta = arma::solve(m_regression_matrix, m_aux_vector);
  }

  void LeastSquares::print() const {
     std::cout << "Model order: " << m_order << std::endl;
     std::cout << "Regression matrix: \n" << m_regression_matrix << std::endl;
     std::cout << "Regression matrix determinant: " << arma::det(m_regression_matrix) << std::endl;
     std::cout << "Auxiliary vector: \n" << m_aux_vector << std::endl;
     std::cout << "Theta: \n" << m_theta << std::endl;
  }

  void LeastSquares::fill_vector(const boost::circular_buffer<double> &phi) {
     if(m_phi.n_rows != phi.size()) throw std::logic_error("Not enough elements in the c_buffer");
     for (unsigned int i = 0; i < phi.size(); i++) {
        m_phi(i) = phi[i];
     }
  }
}
