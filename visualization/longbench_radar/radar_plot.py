import numpy as np
import pdb
import matplotlib.pyplot as plt
# import seaborn as sns # improves plot aesthetics
from longbench_results import LONGBENCH_RESULTS
import math

FONTSIZE = 14
LABELSIZE = 15
LEGENDSIZE = 14
TEXTSIZE = 20
font_config = {"font.size": FONTSIZE} # , "font.family": "Times New Roman"
plt.rcParams.update(font_config)

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])
def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=4):
        angles = np.arange(0, 360, 360./len(variables)) + 30
        axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                             label="axes{}".format(i)) for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=variables, fontsize=25, position=(0.0, 0.25))
        # [txt.set_rotation(angle) for txt, angle
        #  in zip(text, text_angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x * 10)/10)
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                # grid = grid[::-1] # hack to invert grid
                          # gridlabels aren"t reversed
                pass
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[i], fontsize=20, color="gray") # color="gray")
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.angles = angles
        # self.axes = axes

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

# example data
blank = " "
variables = ("Single-Doc QA", "Multi-Doc QA", "Summarization", "Few-shot", "Synthetic", "Code")
# variables = ("1", "2", "2", "2", "2", "2")


def estimate_axis_range(datas, alg_buf):

    data_min = np.array(list(datas)).min(axis=0)
    data_max = np.array(list(datas)).max(axis=0)

    axis_max = data_max
    axis_min = data_min - (data_max - data_min) * 0.2

    if "Baseline" not in alg_buf:
        return [(axis_min[idx], axis_max[idx]) for idx in range(len(axis_min))]

    baseline_value = datas[alg_buf == "Baseline"].squeeze(0)
    poor_index = (data_max > baseline_value)
    # print(axis_max[poor_index])
    # print(data_min[poor_index])
    # print(data_max[poor_index])

    axis_min[poor_index] = data_min[poor_index] - (data_max[poor_index] - data_min[poor_index])*5

    # print(axis_min <= data_min)
    # print(axis_max >= data_max)
    return [(axis_min[idx], axis_max[idx]) for idx in range(len(axis_min))]


def radar_plot(config_name):
    datas = [value for key, value in LONGBENCH_RESULTS[config_name].items()]

    ranges = estimate_axis_range(np.array(list(datas)), np.array(list(LONGBENCH_RESULTS[config_name].keys())))
    # axis_min = np.array(list(datas)).min(axis=0)
    # axis_max = np.array(list(datas)).max(axis=0)
    # # datas = ([31.07333333, 23.95, 26.74666667, 63.74666667, 30.5, 56.885], [30.16666667, 23.24333333, 26.44666667, 63.65666667, 32.25, 55.865])
    # # ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]
    # ranges = [(axis_min[idx] - (axis_max[idx] - axis_min[idx])*0.2, axis_max[idx]) for idx in range(len(datas[0]))]
    # if config_name == "H2O-Mistral-7B":
    #     ranges = [(axis_min[idx] - (axis_max[idx] - axis_min[idx]) * 5, axis_max[idx]) for idx in range(len(datas[0]))]
    # if config_name == "H2O-Llama-3-8B":
    #     ranges = [(axis_min[idx] - (axis_max[idx] - axis_min[idx]) * 5, axis_max[idx]) for idx in range(len(datas[0]))]

    zorder = range(len(datas[0]))
    # datas = ([79.03, 573, .20], [79.13, 82, .31])
    # ranges = [(78.90, 79.20), (650, 50), (0.0, 0.5)]
    # plotting
    color_buf = ["red", "green", "dodgerblue", "darkmagenta", "turquoise", "gray"]
    fig1 = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig1, variables, ranges)
    for idx, data in enumerate(datas):
        if list(LONGBENCH_RESULTS[config_name].keys())[idx] == "Baseline" or list(LONGBENCH_RESULTS[config_name].keys())[idx] == "Llama-3-8B":
            linewidth = 4
        else:
            linewidth = 2

        radar.plot(data, color=color_buf[idx], linewidth=linewidth, markersize=6, marker="o", zorder=zorder[idx])
        radar.fill(data, alpha=0)

    n_legend = len(LONGBENCH_RESULTS[config_name].keys())
    if n_legend == 4:
        legend_loc = (0.75, -0.1)
    elif n_legend == 3:
        legend_loc = (0.75, -0.05)
    elif n_legend == 5:
        legend_loc = (0.75, -0.15)
    elif n_legend == 2:
        legend_loc = (0.5, 0)

    if config_name == "RNN":
        legend_fontsize = 22
    else:
        legend_fontsize = 20

    radar.ax.legend([key for key in LONGBENCH_RESULTS[config_name].keys()],
                    labelspacing=0.1, fontsize=legend_fontsize, loc=legend_loc, frameon=False)
    # plt.subplots_adjust(left=0.01, bottom=0.01, top=0.99, right=0.99, wspace=0.01) # (left=0.125, bottom=0.155, top=0.965, right=0.97, wspace=0.01)

    #### Rotate the variables
    angles = radar.angles / 180 * math.pi
    text_angles = [-60, 0, 60, -60, 0, 60]
    for label, angle, text_angle in zip(radar.ax.get_xticklabels(), angles, text_angles):
        x, y = label.get_position()
        # print(x, y, label)
        # print(axes[0].get)
        lab = radar.ax.text(angle, 0, label.get_text(), fontsize=20, transform=label.get_transform(), ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(text_angle)
    radar.ax.set_xticklabels([])

    # plt.savefig(f"./figure/pdf/radar_{config_name}.pdf", bbox_inches="tight")
    plt.savefig(f"./figure/png/radar_{config_name}.pdf", bbox_inches="tight", dpi=600)
    plt.close()
    # plt.show()

def main():

    # RNN legend too small
    # lingua legend remove llama-3-8B

    config_name_buf = LONGBENCH_RESULTS.keys()
    # config_name_buf = ["RNN"]
    for config_name in config_name_buf:
        print(config_name)
        radar_plot(config_name)




if __name__ == "__main__":

    from matplotlib import pyplot as plt

    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    # t = figure.text(0.5, 0.5, "some text")
    # t.set_rotation(90)
    # labels = ax.get_xticklabels()
    # for label in labels:
    #     label.set_rotation(45)
    # plt.show()
    main()
