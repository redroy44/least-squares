#ifndef LS_H_
#define LS_H_

#include <armadillo>

namespace ls {
  class LeastSquares {
  public:
    LeastSquares();
    virtual ~LeastSquares();
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
