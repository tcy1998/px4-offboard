import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

SIZE = 15
rc('font', **{'family': 'serif', 'serif': ['Open Sans']})
# rc('text', usetex=True)
rc('xtick', labelsize=SIZE)
rc('ytick', labelsize=SIZE)
# matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


class PlotFigs:
    def __init__(self, time_step, sys, esti):
        self.time_k = range(time_step + 1)
        self.sys = sys
        self.esti = esti

    def x1(self):
        plt.plot(self.time_k, self.sys.X1[:], label=r'$x_1$', color='#E64A4A')
        plt.plot(self.time_k, self.esti.Xh1[:], label=r'$\hat{x}_1$', color='black')
        plt.ylabel(r'$x$', fontsize=SIZE)
        plt.xlabel(r'$time[k]$', fontsize=SIZE)
        plt.legend(fontsize=SIZE)
        plt.show()

    def x2(self):
        plt.plot(self.time_k, self.sys.X2[:], label=r'$x_2$', color='#E64A4A')
        plt.plot(self.time_k, self.esti.Xh2[:], label=r'$\hat{x}_2$', color='black')
        plt.ylabel(r'$x$', fontsize=SIZE)
        plt.xlabel(r'$time[k]$', fontsize=SIZE)
        plt.legend(fontsize=SIZE)
        plt.show()

    def x3(self):
        plt.plot(self.time_k, self.sys.X3[:], label=r'$x_3$', color='#E64A4A')
        plt.plot(self.time_k, self.esti.Xh3[:], label=r'$\hat{x}_3$', color='black')
        plt.ylabel(r'$x$', fontsize=SIZE)
        plt.xlabel(r'$time[k]$', fontsize=SIZE)
        plt.legend(fontsize=SIZE)
        plt.show()

    def d(self):
        plt.plot(self.time_k, self.sys.F1[:], label=r'$d_1$', color='#E64A4A')
        plt.plot(self.time_k, self.esti.Fh1[:], label=r'$\hat{d}_1$', color='black')
        plt.ylabel(r'$d$', fontsize=SIZE)
        plt.xlabel(r'$time[k]$', fontsize=SIZE)
        plt.legend(fontsize=SIZE)
        plt.show()
