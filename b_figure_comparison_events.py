"""
    Author: Zy_Zhu
    Date  : 2023-07-12 11:05
    Proj  : b_figure_comparison_events.py
"""
################################################################################################
# There are two ways to draw a half violin: 1.ptitprince 2.matplotlib
# A whole picture
################################################################################################
import pickle
import time
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import ptitprince as pt

warnings.filterwarnings('ignore')


class Path:
    pickle_dir = 'dur_area_per_dict.pickle'


def custom_parameter():
    # 自定义各种参数
    rcParams['savefig.dpi'] = 600
    rcParams['font.family'] = 'Arial'
    # rcParams['savefig.facecolor'] = 'None'
    # rcParams['savefig.edgecolor'] = 'None'
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.01
    rcParams['font.size'] = 18
    # rcParams['savefig.transparent'] = True


def read_data(start=1, end=17):
    data = pickle.load(open(Path.pickle_dir, 'rb'))
    duration = data['duration_list'][start - 1:end]
    area_per = data['area_per_list'][start - 1:end]

    # There was a bug earlier where the two lines of code were transposed in sequence, which caused the duration maximum to be deleted and then changed when indexing the duration maximum
    area_per[0].pop(duration[0].index(np.max(duration[0])))  # Deletes the affected area for the location of the event with the longest duration
    duration[0].remove(np.max(duration[0]))  # Delete the longest drought event in the Amazon

    return duration, area_per


def add_canvas():
    fig, ax = plt.subplots(nrows=2, ncols=1,
                           figsize=(24, 14), gridspec_kw={'wspace': 0.15, 'hspace': 0.1})

    return fig, ax


def draw_half_violin(ax, duration, color='#4797C8', offset=0.32, width=0.5):
    #ptitprince drawing of half a violin
    pt.half_violinplot(data=duration,
                       color=color,
                       bw=.2, cut=0,
                       scale="area", width=0.7, inner=None, orient='v',
                       linewidth=0,
                       offset=offset,
                       alpha=0.8,
                       ax=ax
                       )
    # matplotlib drawing of half a violin
    # vp = ax.violinplot(duration, widths=width,
    #                    positions=[i - offset for i in range(17)],
    #                    # points=500,
    #                    showmeans=False, showextrema=False, showmedians=False, vert=True
    #                    )
    #
    # for idx, b in enumerate(vp['bodies']):
    #     m = np.mean(b.get_paths()[0].vertices[:, 0])
    #     b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    #     b.set_color(color)
    #     b.set_alpha(0.8)
    #     b.set_edgecolor(None)


def define_cmap():
    cmap = mpl.colormaps['GnBu']
    colors = [cmap(i) for i in np.linspace(0.25, 1, 100)]
    new_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return new_cmap


def draw_scatter(ax, duration, area_per, new_cmap):
    scatter_bar = None
    for i, dur in enumerate(duration):
        x_jitter = np.random.normal(loc=i, scale=0.07, size=len(dur))
        bar = ax.scatter(x_jitter, dur, s=10, alpha=0.7, c=area_per[i], cmap=new_cmap, vmin=0, vmax=1, linewidths=0)
        if i == 0:  # Randomly select a return for subsequent drawing of the colorbar
            scatter_bar = bar
    return scatter_bar


def colorbar_and_param(fig, ax, bar):
    cb = fig.colorbar(mappable=bar,
                      ax=ax,
                      shrink=1,  
                      # pad=0.025, 
                      pad=0.05,
                      orientation='vertical',
                      aspect=20,
                      )
    cb.outline.set_linewidth(0)
    cb.ax.tick_params(which='both', length=0)
    cb.ax.set_yticks(np.arange(0, 1 + 0.1, 0.2), np.arange(0, 100 + 1, 20))


def draw_box(ax, duration, color='#4797C8', position=np.arange(17) - 0.1):
    ax.boxplot(duration, showfliers=False, widths=0.15,
               boxprops={'linewidth': 1, 'color': color, 'facecolor': 'none'},
               medianprops={"color": color, "linewidth": 1}, 
               whiskerprops={"color": color, "linewidth": 1}, 
               capprops={"color": color, "linewidth": 1},
               patch_artist=True,
               positions=position)


def draw_ax2(ax, area_per, color='#F07980'):
    ax2 = ax.twinx()

    draw_half_violin(ax2, area_per, color=color, offset=0.26, width=0.3)
    draw_box(ax2, area_per, color=color, position=np.arange(17) + 0.1)

    return ax2


def add_num(ax, ax2):
    # This is already ax2, so plt.ylim() gets the Y-axis range of ax2
    x_min, x_max = ax2.get_xlim()
    y_min, y_max = ax2.get_ylim()
    print(x_min, x_max, y_min, y_max)
    x_len = x_max - x_min
    y_len = y_max - y_min
    y_one_x = ((1 / x_len) / (7 / 25)) * y_len
    square_width = 0.3  # Width of a square
    square_height = y_one_x * square_width

    # Create a rounded square
    for i in range(17):
        square = patches.FancyBboxPatch((i - 0.15, -y_one_x * 0.43),
                                        width=square_width, height=square_height * 0.6,
                                        linewidth=1,
                                        edgecolor='#595959', facecolor='#9DC3E6',
                                        # boxstyle='round,pad=0.4,rounding_size=1',
                                        mutation_scale=0.2,
                                        mutation_aspect=y_one_x,
                                        zorder=0,
                                        transform=ax2.transData,
                                        clip_on=False)
        # Add a square to the coordinate system
        ax.add_patch(square)


def set_other_parma(ax1, ax2, x_min, x_max):
    ax1_color = '#4797C8'
    ax2_color = '#F07980'

    ax1.tick_params(axis='y', which='major', color=ax1_color, labelcolor=ax1_color, labelsize=18)
    ax2.tick_params(axis='y', which='major', color=ax2_color, labelcolor=ax2_color, labelsize=18)

    ax2.spines['left'].set_color(ax1_color)
    ax2.spines['right'].set_color(ax2_color)


    ax1.set_ylabel('Drought duration', color=ax1_color, fontsize=20, labelpad=15)
    ax2.set_ylabel('Drought area (%)', color=ax2_color, rotation=-90, fontsize=20, labelpad=18)

    ax1.tick_params(axis='x', length=0, labelsize=14, which='major', pad=10)
    ax1.set_xlim(-0.8, 16.5)
    ax1.set_xticks(np.arange(17))
    ax1.set_xticklabels([f"{i}" for i in range(x_min, x_max + 1)], fontsize=18)


    # ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_yticks(np.arange(0, 1 + 0.2, 0.2), np.arange(0, 100 + 1, 20))
    ax2.set_ylim(-0.032142386604260234, 1.0491496374573506)


def draw_one(fig, ax, a, b):
    duration, area_per = read_data(a, b) 
    draw_half_violin(ax, duration, width=0.6)
    cmap = define_cmap()  
    bar = draw_scatter(ax, duration, area_per, cmap) 
    colorbar_and_param(fig, ax, bar) 
    draw_box(ax, duration) 
    ax2 = draw_ax2(ax, area_per) 
    add_num(ax, ax2) 
    set_other_parma(ax, ax2, x_min=a, x_max=b) 


def main():

    custom_parameter()
    fig, ax = add_canvas() 

    draw_one(fig, ax[0], 1, 17)
    draw_one(fig, ax[1], 18, 34)
    # ax[0].set_xlim(-0.8, 16.5)
    # ax[0].set_xticks(np.arange(17))
    # ax[0].set_xticklabels([f"{i}" for i in range(1, 18)], fontsize=14)

    # plt.savefig(f'b_figure_comparison_events_{a}_{b}.png')
    plt.savefig(f'b_figure_comparison_events_ptitprince.png')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start, '秒')  # 5.3s
