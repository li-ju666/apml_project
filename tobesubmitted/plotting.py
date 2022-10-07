from matplotlib import pyplot
import pandas as pd
from matplotlib.pyplot import figure
import numpy as np

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors]


def plot_skill(data, img_name):
    data = data[data[:, 1].argsort()]
    
    figure(figsize=(8, 3))
    
    color1 = "#abdbe3"
    color2 = "#063970"
    
    pyplot.bar(data[:, 0], data[:, 1], yerr=data[:, 2], alpha=0.5, align='center',
               color=get_color_gradient(color1, color2, len(data[:, 0])), edgecolor="black",
               error_kw=dict(ecolor='red', alpha=0.5))
    pyplot.xticks(rotation=90)
    pyplot.ylabel("Skill")
    pyplot.title("Skills of Teams")
    
    # Save the figure and show
    pyplot.tight_layout()
    pyplot.savefig(img_name, dpi=500)

