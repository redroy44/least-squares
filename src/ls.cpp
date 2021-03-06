#include "../include/ls.h"
#include <iostream>

namespace ls {
   const arma::vec& IEstimator::getTheta() const {
      return m_theta;
   }

   const double& IEstimator::getNoiseVar() const {
      return m_driving_noise;
   }

   void IEstimator::print() const {
      std::cout << "Theta: \n" << m_theta << std::endl;
      std::cout << "Driving noise variance: \n" << m_driving_noise << std::endl;
   }

   LeastSquares::LeastSquares(std::string s) {
      std::cout << s << std::endl;
   }

   LeastSquares::LeastSquares(const arma::vec &signal, const unsigned int &order) : LeastSquares("LS offline estimation") {
      m_signal = signal;
      m_order = order;
      m_regression_matrix = arma::zeros(order, order);
      m_phi = arma::zeros(order);
      m_theta = arma::zeros(order);
      m_aux_vector = arma::zeros(order);
      m_driving_noise = 0;
      m_signal_length = signal.n_rows;
      m_sum_error = 0;
   }

   LeastSquares::LeastSquares(const unsigned int &order) : LeastSquares("LS online estimation") {
      m_order = order;
      m_regression_matrix = arma::zeros(order, order);
      m_phi = arma::zeros(order);
      m_theta = arma::zeros(order);
      m_aux_vector = arma::zeros(order);
      m_driving_noise = 0;
      m_signal_length = 0;
      m_sum_error = 0;
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
            m_sum_error += as_scalar(arma::square(m_signal(i + 1) - m_phi.t()*m_theta));
         }
      }
      m_theta = arma::solve(m_regression_matrix, m_aux_vector);
      // m_driving_noise = m_sum_error/double(m_signal_length);
      // WA hardcode this for now
      m_driving_noise = 0.01;
   }

   void LeastSquares::estimate(const arma::vec &phi, const double &sample) {
      m_signal_length++;
      m_phi = phi;
      m_regression_matrix += m_phi * m_phi.t();
      if(arma::det(m_regression_matrix) < 1)
         std::cout << "det A: " << arma::det(m_regression_matrix) << std::endl;
      m_aux_vector += sample * m_phi;
      m_theta = arma::solve(m_regression_matrix, m_aux_vector);
      m_sum_error += as_scalar(arma::square(sample - m_phi.t()*m_theta));
      m_driving_noise = m_sum_error/double(m_signal_length);
   }

   void LeastSquares::print() const {
      std::cout << "Model order: " << m_order << std::endl;
      std::cout << "Regression matrix: \n" << m_regression_matrix << std::endl;
      std::cout << "Regression matrix determinant: " << arma::det(m_regression_matrix) << std::endl;
      std::cout << "Auxiliary vector: \n" << m_aux_vector << std::endl;
      std::cout << "Theta: \n" << m_theta << std::endl;
      std::cout << "Driving noise variance: \n" << m_driving_noise << std::endl;
   }
}
