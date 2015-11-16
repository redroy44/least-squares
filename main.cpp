#include <iostream>
#include "ls.h"
#include <armadillo>
#include <boost/circular_buffer.hpp>

using namespace arma;

int main () {
arma_rng::set_seed(666);

// generate signal
unsigned int order = 2;
unsigned int length = 1000;
vec signal = ones(length + order);
for(unsigned int i = order; i < signal.n_rows; i++) {
   signal(i) = 1.02*signal(i-1) - 0.53*signal(i-2) + as_scalar(randn(1));
}
signal = signal.rows(order, signal.n_rows-1);
// offline estimation
ls::LeastSquares obj = ls::LeastSquares(signal, order);
obj.estimate();
obj.print();

//online estimation
boost::circular_buffer<double> cb(order);
// initialize with random values
for(unsigned int i = 0; i < order; i++) {
   cb.push_front(signal(i));
}
ls::LeastSquares obj2 = ls::LeastSquares(order);

for(unsigned int i = order; i < signal.n_rows; i++) {
   obj2.estimate(cb, signal(i));
   cb.push_front(signal(i));
}
obj2.print();

return 0;
}
