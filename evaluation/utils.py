import matplotlib

FIGURE_PATH = ""

def plot(PRINT_PLOTS, name, plt):
  if PRINT_PLOTS:
    plt.savefig(FIGURE_PATH+"powerlaw.eps", format="eps")
    
