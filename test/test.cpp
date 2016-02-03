//#include <iostream>
#include "ls.h"
#include <armadillo>
#include <boost/circular_buffer.hpp>
#include "gtest/gtest.h"

using namespace arma;

TEST(DummyTest, Positive) {
   EXPECT_EQ(1, 1);
}

TEST(LeastSquares, Offline) {
   unsigned int order = 2;
   unsigned int length = 1000;
   vec signal = ones(length + order);
   ASSERT_TRUE(signal.load("../test/data/signal.dat", raw_ascii));
   ASSERT_EQ(1000u, signal.n_rows);
   ls::LeastSquares obj = ls::LeastSquares(signal, order);
   EXPECT_NO_THROW(obj.estimate());
   vec theta_est = obj.getTheta();
   EXPECT_NEAR(0.25, theta_est(0), 0.5);
   EXPECT_NEAR(0.7, theta_est(1), 0.5);
   EXPECT_NEAR(0.1, sqrt(obj.getNoiseVar()), 0.05);
}

TEST(LeastSquares, Online) {
   unsigned int order = 2;
   unsigned int length = 1000;
   vec signal = ones(length + order);
   ASSERT_TRUE(signal.load("../test/data/signal.dat", raw_ascii));
   ASSERT_EQ(1000u, signal.n_rows);
   ls::LeastSquares obj = ls::LeastSquares(order);

   boost::circular_buffer<double> cb(order);
   // initialize
   for(unsigned int i = 0; i < order; i++) {
      cb.push_front(signal(i));
   }

   for(unsigned int i = order; i < signal.n_rows; i++) {
      ASSERT_NO_THROW(obj.estimate(std::vector<double>(cb.begin(), cb.end()), signal(i)));
      cb.push_front(signal(i));
   }
   vec signal_est = vec(signal);
   vec theta_est = obj.getTheta();
   EXPECT_NEAR(0.25, theta_est(0), 0.5);
   EXPECT_NEAR(0.7, theta_est(1), 0.5);
   EXPECT_NEAR(0.1, sqrt(obj.getNoiseVar()), 0.05);

   for(unsigned int i = order; i < signal_est.n_rows; i++) {
      signal_est(i) = theta_est(0)*signal_est(i-1) + theta_est(1)*signal_est(i-2) + sqrt(obj.getNoiseVar())*as_scalar(randn(1));
   }
   ASSERT_EQ(1000u, signal_est.n_rows);
   ASSERT_TRUE(signal_est.save("signal_est_LS.dat", raw_ascii));
}

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from test.cpp\n");
  ::testing::InitGoogleTest(&argc, argv);
  arma_rng::set_seed(666);
  std::ostream nullstream(0);
  set_stream_err2(nullstream);
  return RUN_ALL_TESTS();
}
