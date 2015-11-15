#include <iostream>
#include "ls.h"
#include <armadillo>
#include <boost/circular_buffer.hpp>

using namespace arma;

int main () {
arma_rng::set_seed(0);
boost::circular_buffer<int> cb(3);

// generate signal
vec signal = linspace<vec>(0, 2*datum::pi, 100);
unsigned int order = 4;
//signal.print();
signal.transform([](double val) {return (0.3*sin(val)+0.7*cos(val)+0.5*as_scalar(randu(1)));});
//signal.print();

ls::LeastSquares obj = ls::LeastSquares(signal, order);
obj.estimate();
obj.print();

return 0;
}
