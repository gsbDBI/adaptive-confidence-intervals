
"""
This script contains helper functions to make plots presented in our paper Confidence Intervals for Policy Evaluation in Adaptive Experiments (https://arxiv.org/abs/1911.02768)
"""

import seaborn as sns
from adaptive_CI.compute import collect
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from time import time
from glob import glob
import scipy.stats as stats
from IPython.display import display, HTML
from adaptive_CI.saving import *
import pickle
from pickle import UnpicklingError
import copy
from itertools import compress
import scipy.stats as stats
import os

def plot_contrast(df,
                  row_order=['nosignal', 'lowSNR', 'highSNR'],
                  col_order=['mse', 'bias', 'CI_width', '90% coverage of t-stat'],
                  col_names=['RMSE', 'bias', 'Confidence Interval Radius', '90% coverage'],
                   hue_order=['uniform', 'lvdl', 'two_point'],
                   labels=['uniform', 'constant allocation rate', 'two-point'],
                  name=None):
    """
    Plot RMSE, bias and 90% coverage of t-statisitcs in cases of no-signal and high-SNR across different weighting schemes.
    """
    palette = sns.color_palette("muted")[:len(hue_order)]
    g = sns.catplot(x="T",
                    y="value",
                    col="statistic",
                    col_order=col_order,
                    row="dgp",
                    row_order=row_order,
                    hue='method',
                    hue_order=hue_order,
                    palette=palette,
                    kind="point",
                    sharex=False,
                    sharey=False,
                    legend=False,
                    legend_out=True,
                    margin_titles=True,
                    data=df)

    # Plot ROOT mse
    for i, dgp in enumerate(row_order):
        g.axes[i, 0].clear()
        sns.pointplot(x='T',
                      y="value",
                      hue='method',
                      hue_order=hue_order,
                      palette=palette,
                      ax=g.axes[i, 0],
                      data=df.query(f"statistic=='mse' & dgp=='{dgp}'"),
                      estimator=lambda x: np.sqrt(np.mean(x)),
                      markers="")
        g.axes[i, 0].get_legend().remove()
    g.axes[0, 0].set_xlabel("")
    g.axes[0, 0].set_ylabel("")


    # Add row and column names
    g.row_names = ['NO SIGNAL', 'LOW-SIGNAL', 'HIGH SNR']
    g.col_names = col_names

    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes[:, 1]:
        ax.axhline(0, color="black", linestyle='--')

    # CI radius
    for ax in g.axes[:, 2]:
        ax.set_ylim((0, ax.get_ylim()[1]))

    for ax in g.axes[:, -1]:
        ax.axhline(0.90, color="black", linestyle='--')

    g.fig.tight_layout()

    # Add legend
    handles, ls = g._legend_data.values(), g._legend_data.keys()
    label_dict = dict(zip(hue_order, labels))
    labels = [label_dict[k] for k in ls]
    g.fig.legend(labels=labels, frameon=False,
                 handles=handles, loc='lower center',
                 ncol=len(labels), bbox_to_anchor=(0.5, 0.0))

    g.set_xlabels("")
    g.set_ylabels("")
    g.fig.tight_layout()

    g.fig.subplots_adjust(bottom=0.1)

    if name is not None:
        g.fig.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    plt.show()

def plot_arm_values(df,
                   col_order=['mse', 'bias', 'CI_width', '90% coverage of t-stat'],
                   row_order=[2, 0],
                   hue='method',
                   hue_order=['uniform', 'lvdl', 'two_point'],
                   labels=['uniform', 'constant allocation rate', 'two-point'],
                   noise_func='uniform',
                   name=None):
    """
    Plot converged RMSE and bias of bad arm and good arm across different weighting schemes.
    """
    palette = sns.color_palette("muted")[:len(hue_order)]

    order = ['nosignal', 'lowSNR', f'highSNR']
    order_name = ['NO SIGNAL', 'LOW SNR', 'HIGH SNR']
    g = sns.catplot(x="dgp",
                    y="value",
                    order=order,
                    hue='method',
                    hue_order=hue_order,
                    palette=palette,
                    col="statistic",
                    col_order=col_order,
                    row="policy",
                    row_order=row_order,
                    kind="point",
                    sharex=False,
                    sharey=False, #'col',
                    legend=False,
                    legend_out=True,
                    margin_titles=True,
                    data=df)

    # Plot RMSE of bad arm based on MSE
    for i, arm in enumerate(row_order):
        g.axes[i, 0].clear()
        sns.pointplot(x='dgp',
                      y="value",
                      order=order,
                      hue='method',
                      hue_order=hue_order,
                      palette=palette,
                      ax=g.axes[i, 0],
                      data=df.query(f"policy=={arm} & statistic=='mse'"),
                      estimator=lambda x: np.sqrt(np.mean(x)),
                      )
        g.axes[i, 0].get_legend().remove()
        g.axes[i, 0].set_xlabel("")
        g.axes[i, 0].set_ylabel("")


    # Add row and column names
    g.col_names = ['RMSE', 'Bias', 'Confidence Interval Radius', '90% coverage']
    g.row_names = ['GOOD ARM', 'BAD ARM']

    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Set xticklabels to be [NO SIGNAL, LOW SIGNAL and HIGH SIGNAL]
    for ax in g.axes.flat:
        ax.set_xticklabels(order_name)

    # Bias
    for ax in g.axes[:, 1]:
        ax.axhline(0, color="black", linestyle='--')

    # CI radius
    for ax in g.axes[:, 2]:
        ax.set_ylim((0, ax.get_ylim()[1]))

    # Coverage
    for ax in g.axes[:, -1]:
        ax.axhline(0.90, color="black", linestyle='--')

    # Add legend
    handles, ls = g._legend_data.values(), g._legend_data.keys()
    label_dict = dict(zip(hue_order, labels))
    labels = [label_dict[k] for k in ls]
    g.fig.legend(labels=labels, frameon=False,
                 handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.01))

    g.set_xlabels("")
    g.set_ylabels("")

    g.fig.tight_layout()
    g.fig.subplots_adjust(bottom=0.1)
    if name is not None:
        plt.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    plt.show()


def plot_hist(df,
              methods = ['bernstein', 'uniform', 'lvdl', 'two_point'],
              col_names=['Sample mean', 'Uniform', 'Constant allocation', 'Two-point allocation'],
              name=None):
    """
    Plot histogram of normalized errors: relative error normalized by Monte Carlo standard deviation, and t-statistics of CLT.
    """
    assert df['noise_func'].nunique() == 1
    assert df['floor_decay'].nunique() == 1
    assert df['initial'].nunique() == 1

    g = sns.FacetGrid(col="method",
                      row='dgp',
                      row_order=[f'nosignal', f'lowSNR', f'highSNR'],
                      hue="statistic",
                      hue_order=['t-stat'],
                      col_order=methods,
                      legend_out=True,
                      sharex=False,
                      sharey=True,
                      margin_titles=True,
                      data=df)


    with warnings.catch_warnings():
        # Block an annoying but inconsequential warning from seaborn v0.11.
        warnings.simplefilter(action='ignore', category=FutureWarning)
        g = g.map(sns.distplot, "value", hist=False, kde=True)

    # Add row and column names
    g.row_names = ['NO SIGNAL', 'LOW SNR', 'HIGH SNR']
    g.col_names = col_names

    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Add histogram of N(0, 1)
    xs = np.linspace(-3, 3)
    for ax in g.axes.flatten():
        ax.plot(xs, norm.pdf(xs), label='N(0,1)', **
                {"color": "black", "linestyle": "--", "linewidth": 2})
        ax.set_xticks([-2, 0, 2])
        ax.set_xlim([-3, 3])

    # Add legend
    handles, labels = g._legend_data.values(), g._legend_data.keys()
    g.fig.legend(labels=['Studentized statistic', 'N(0,1)'],
                 loc='center',  ncol=3, bbox_to_anchor=(0.5, 0.03))
    g.set_xlabels("")
    g.set_ylabels("")

    g.fig.tight_layout()
    g.fig.subplots_adjust(bottom=0.1)
    if name is not None:
        plt.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    plt.show()


def plot_wdecorr_comparison(df):
    """
    Compare W-decorrelation and two-point allocation rate. Plot RMSE and 90% coverage of t-stat of good arm high SNR, bad arm high SNR and arms no signal.
    """
    hue_order = ['W-decorrelation_15', 'two_point']
    palette = sns.color_palette("muted")[:len(hue_order)]
    col_order = ['highSNR:0', 'highSNR:2', 'nosignal:0']

    df_subset = df.query('method == @hue_order')
    df_subset['experiment_policy'] = df_subset['dgp'] + ":" + df_subset['policy'].astype(str)

    g = sns.catplot(x="T",
                    y="value",
                    hue='method',
                    hue_order=hue_order,
                    palette=palette,
                    col="experiment_policy",
                    col_order=col_order,
                    row="statistic",
                    row_order=['mse', 'CI_width', '90% coverage of t-stat'],
                    kind="point",
                    sharex=True,
                    sharey='row',
                    legend=False,
                    legend_out=True,
                    margin_titles=True,
                    data=df_subset)


    # plot RMSE of good arm high SNR
    for i, col in enumerate(col_order): # good arm, bad arm, bad(any) arm
        g.axes[0, i].clear()
        sns.pointplot(x='T',
                      y="value",
                      hue='method',
                      hue_order=hue_order,
                      palette=palette,
                      ax=g.axes[0, i],
                      data=df_subset.query(f"statistic=='mse' & experiment_policy == @col"),
                      estimator=lambda x: np.sqrt(np.mean(x)))
        g.axes[0, i].get_legend().remove()
        g.axes[0, i].set_xlabel("")
        g.axes[0, i].set_ylabel("")


    # Add row and column names
    g.row_names = ['RMSE', 'Confidence Interval Radius', '90% coverage of t-stat']
    g.col_names = ['GOOD ARM: HIGH SIGNAL', 'BAD ARM: HIGH SIGNAL', 'NO SIGNAL']


    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # rmse
    for ax in g.axes[0, :]:
        ax.axhline(0., color="black", linestyle='--')
    
    # ci radius
    for ax in g.axes[1, :]:
        ax.axhline(0., color="black", linestyle='--')
    
    # coverage
    for ax in g.axes[2, :]:
        ax.axhline(.9, color="black", linestyle='--')

    # Add legend
    handles, labels = g._legend_data.values(), g._legend_data.keys()
    g.fig.legend(labels=['W-decorrelation', 'Two-point allocation'],
                 handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))

    g.set_xlabels("Experiment length")
    g.set_ylabels("")

    g.fig.tight_layout()
    g.fig.subplots_adjust(bottom=0.1)
    
    return g.fig

def plot_lambda(df_lambdas, save=False):
    """
    Plot lambda(T-t) for stablevar-typed weights.
    """
    assert df_lambdas['T'].nunique() == 1
    save = False
    g = sns.relplot(x="time",
                    y="value",
                    col="dgp",
                    col_order=[f'nosignal', f'lowSNR', f'highSNR'],
                    kind="line",
                    row='policy',
                    row_order=[2, 0],
                    facet_kws=dict(
                        sharex=False,
                        sharey=False),
                    legend=False,
                    ci=None,
                     data=df_lambdas)

    g.col_names = ['NO SIGNAL', 'LOW SIGNAL', 'HIGH SIGNAL']
    g.row_names = ['GOOD ARM', 'BAD ARM']

    Tmax = df_lambdas['T'].max()
    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
        ax.set_yscale('log')
        xticks = ax.get_xticks()
        ax.set_xticks([0, xticks[len(xticks)//2], xticks[-1]])
        ax.axhline(1.0, color="black", linestyle='--')
        ax.set_xticklabels([0, Tmax//2, Tmax])
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    g.set_xlabels("")
    g.set_ylabels("")

    g.fig.tight_layout()
    if save:
        plt.savefig(f'figures/lambdas.pdf', bbox_inches='tight')
        plt.show()


