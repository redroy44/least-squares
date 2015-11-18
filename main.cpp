#include <iostream>
#include "ls.h"
#include <armadillo>
#include <boost/circular_buffer.hpp>

using namespace arma;

int main () {
arma_rng::set_seed(666);
std::ostream nullstream(0);
set_stream_err2(nullstream);

// generate signal
unsigned int order = 2;
unsigned int length = 1000;
vec signal = ones(length + order);
for(unsigned int i = order; i < signal.n_rows; i++) {
   signal(i) = 0.7*signal(i-1) + 0.25*signal(i-2) + 0.1*as_scalar(randn(1));
}
signal = signal.rows(order, signal.n_rows-1);
signal.save("signal.dat", raw_ascii);

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
   try {
      obj2.estimate(cb, signal(i));
   }
   catch(std::runtime_error e) {
      std::cout << "Exception in online LS: " << e.what() << std::endl;
      break;
   }
   cb.push_front(signal(i));
}
obj2.print();

vec signal_est = ones(length + order);
vec theta_est = obj.getTheta();
for(unsigned int i = order; i < signal.n_rows; i++) {
   signal_est(i) = theta_est(0)*signal_est(i-1) + theta_est(1)*signal_est(i-2);
}
signal_est = signal_est.rows(order, signal_est.n_rows-1);
signal_est.save("signal_est_LS.dat", raw_ascii);

return 0;
}
