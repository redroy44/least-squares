#include <iostream>
#include "ls.h"
#include <armadillo>
#include <boost/circular_buffer.hpp>

using namespace arma;

int main () {
arma_rng::set_seed(0);
boost::circular_buffer<int> cb(3);

// generate signal
vec signal = ones(1000);
unsigned int order = 2;
for(unsigned int i = order; i < signal.n_rows; i++) {
   signal(i) = 1.02*signal(i-1) - 0.53*signal(i-2) + as_scalar(randn(1));
}
signal.print();
//signal.transform([](double val) {return (0.3*sin(val)+0.7*cos(val)+0.5*as_scalar(randu(1)));});
//signal.print();

ls::LeastSquares obj = ls::LeastSquares(signal, order);
obj.estimate();
obj.print();

return 0;
}
