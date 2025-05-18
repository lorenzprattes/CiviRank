import matplotlib

FIGURE_PATH = ""

def plot(PRINT_PLOTS, name, plt):
  if PRINT_PLOTS:
    plt.tight_layout()
    plt.savefig(FIGURE_PATH+ name + ".eps", format="eps", transparent=False, bbox_inches="tight")
    plt.show()
