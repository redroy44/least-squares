#ifndef LS_H_
#define LS_H_

#include <armadillo>
#include <boost/circular_buffer.hpp>
#include <exception>
#include <string>

namespace ls {
   class IEstimator {
   public:
      virtual ~IEstimator();

      virtual void estimate() = 0;
      virtual void estimate(const arma::vec &, const double &) = 0;

   protected:
      arma::vec m_signal;
      arma::vec m_theta;
      arma::vec m_phi;
      double m_driving_noise;
      unsigned int m_signal_length;
   };

  class LeastSquares {
  public:
    LeastSquares(std::string);
    LeastSquares(const arma::vec &signal, const unsigned int &order);
    LeastSquares(const unsigned int &order);
    virtual ~LeastSquares();

    const arma::vec& getTheta() const;
    const double& getNoiseVar() const;

    void estimate();
    void estimate(const arma::vec &, const double &);
    void print() const;

  private:
    arma::vec m_signal;
    unsigned int m_order;

    arma::mat m_regression_matrix;
    arma::vec m_theta;
    arma::vec m_phi;
    arma::vec m_aux_vector;
    double m_driving_noise;
    double m_sum_error;
    unsigned int m_signal_length;

  };

  class RecursiveLeastSquares {
  public:
    RecursiveLeastSquares();
    virtual ~RecursiveLeastSquares();
    void printMat() const;

  private:
    arma::mat matrix;
  };
}

#endif /* LS_H_ */
