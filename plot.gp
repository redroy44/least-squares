set title "Signals plot"
set xlabel "Sample"
set ylabel "Amplitude"
set terminal jpeg size 1280,720
set output "signal_plot.jpg"
plot '../test/data/signal.dat' with lines lt 1 lc 1 title "Signal", 'signal_est_LS.dat' with lines lt 1 lc 2 title "Estimated Signal"
