import matplotlib.pyplot as plt


def set_context(mode="paper", fontfamily="arial", fontweight="bold"):
    # plt.rc("axes.spines", left=False, right=False, top=False, bottom=False)
    if mode == "paper":
        plt.rc('axes', linewidth=1.5)
        plt.rc('axes', titlesize=15, titleweight=fontweight)
        plt.rc('axes', labelsize=15, labelweight=fontweight)
        font = {
            'family' : fontfamily,
            'weight' : fontweight,
            'size'   : 10}

    if mode == "notebook":
        plt.rc('axes', linewidth=2.5)
        plt.rc('axes', titlesize=20, titleweight=fontweight)
        plt.rc('axes', labelsize=20, labelweight=fontweight)
        font = {
            'family' : fontfamily,
            'weight' : fontweight,
            'size'   : 15}

    if mode == "poster":
        plt.rc('axes', linewidth=3.5)
        plt.rc('axes', titlesize=25, titleweight=fontweight)
        plt.rc('axes', labelsize=25, labelweight=fontweight)
        font = {
            'family' : fontfamily,
            'weight' : fontweight,
            'size'   : 20}

    plt.rc('font', **font)
