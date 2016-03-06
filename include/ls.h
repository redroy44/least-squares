#ifndef LS_H_
#define LS_H_

#include <armadillo>
#include <boost/circular_buffer.hpp>
#include <exception>
#include <string>

namespace ls {
   class IEstimator {
   public:
      virtual ~IEstimator() {};

      virtual void estimate() = 0;
      virtual void estimate(const arma::vec &, const double &) = 0;

      virtual const arma::vec& getTheta() const;
      virtual const double& getNoiseVar() const;
      virtual void print() const;

   protected:
      arma::vec m_signal;
      arma::vec m_theta;
      arma::vec m_phi;
      double m_driving_noise;
      unsigned int m_signal_length;
   };

  class LeastSquares : public IEstimator {
  public:
    LeastSquares(const arma::vec &signal, const unsigned int &order);
    LeastSquares(const unsigned int &order);
    virtual ~LeastSquares() {};

    void estimate();
    void estimate(const arma::vec &, const double &);
    void print() const;

  private:
    LeastSquares(std::string);
    unsigned int m_order;

    arma::mat m_regression_matrix;
    arma::vec m_aux_vector;
    double m_sum_error;

  };

  class RecursiveLeastSquares : public IEstimator {
  public:
    RecursiveLeastSquares(std::string);
    RecursiveLeastSquares(const unsigned int &order);
    virtual ~RecursiveLeastSquares() {};

    void estimate();
    void estimate(const arma::vec &, const double &);
    void print() const;

  private:
    unsigned int m_order;

    arma::mat m_inv_regression_matrix;
    arma::vec m_aux_vector;
    double m_error;
  };
}

#endif /* LS_H_ */
