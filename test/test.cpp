#include <iostream>
#include "ls.h"
#include <armadillo>
#include <boost/circular_buffer.hpp>
#include "gtest/gtest.h"

using namespace arma;

void dupa () {
arma_rng::set_seed(666);
std::ostream nullstream(0);
set_stream_err2(nullstream);

// generate signal
unsigned int order = 2;
unsigned int length = 1000;
// signal(i) = 0.25*signal(i-1) + 0.7*signal(i-2) + 0.1*as_scalar(randn(1));

vec signal = ones(length + order);

if(signal.load("../test/data/signal.dat", raw_ascii)){};
signal.print();

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
vec theta_est = obj2.getTheta();
for(unsigned int i = order; i < signal_est.n_rows; i++) {
   signal_est(i) = theta_est(0)*signal_est(i-1) + theta_est(1)*signal_est(i-2) + sqrt(obj2.getNoiseVar())*as_scalar(randn(1));
}
signal_est = signal_est.rows(order, signal_est.n_rows-1);
signal_est.save("signal_est_LS.dat", raw_ascii);

}

TEST(DummyTest, Positive) {
  EXPECT_EQ(1, 1);
}

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from test.cpp\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
