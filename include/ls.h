#ifndef LS_H_
#define LS_H_

#include <armadillo>

namespace ls {
  class LeastSquares {
  public:
    LeastSquares();
    LeastSquares(const arma::vec &signal, const unsigned int &order);
    virtual ~LeastSquares();

    void estimate();
    void print() const;

  private:
    arma::vec m_signal;
    unsigned int m_order;

    arma::mat m_regression_matrix;
    arma::vec m_theta;
    arma::vec m_phi;
    arma::vec m_aux_vector;

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
