import logging
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sea
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.stats import t

from moabb.analysis.meta_analysis import collapse_session_scores
from moabb.analysis.meta_analysis import combine_effects, combine_pvalues


PIPELINE_PALETTE = sea.color_palette("husl", 6)
sea.set(font='serif', style='whitegrid', palette=PIPELINE_PALETTE)

log = logging.getLogger()


def _simplify_names(x):
    if len(x) > 10:
        return x.split(' ')[0]
    else:
        return x


def score_plot(data, pipelines=None, fig_ax=None):
    '''
    In:
        data: output of Results.to_dataframe()
        pipelines: list of string|None, pipelines to include in this plot
    Out:
        ax: pyplot Axes reference
    '''
    data = collapse_session_scores(data)
    data['dataset'] = data['dataset'].apply(_simplify_names)
    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]
    if fig_ax is None:
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
    else:
        ax = fig_ax
    # markers = ['o', '8', 's', 'p', '+', 'x', 'D', 'd', '>', '<', '^']
    sea.stripplot(data=data, y="dataset", x="score", jitter=0.15,
                  palette=PIPELINE_PALETTE, hue='pipeline', dodge=True, ax=ax,
                  alpha=0.7)
    ax.set_xlim([0, 1])
    ax.axvline(0.5, linestyle='--', color='k', linewidth=2)
    ax.set_title('Scores per dataset and algorithm')
    handles, labels = ax.get_legend_handles_labels()
    color_dict = {l: h.get_facecolor()[0] for l, h in zip(labels, handles)}
    plt.tight_layout()
    if fig_ax is None:
        return fig, color_dict
    else:
        return ax, color_dict


def paired_plot(data, alg1, alg2, fig_ax=None, P=None, T=None, task=None):
    '''
    returns figure with an axis that has a paired plot on it
    Data: dataframe from Results
    alg1: name of a member of column data.pipeline
    alg2: name of a member of column data.pipeline

    '''
    import pdb
    if P is not None:
        p_values = P.T['RE'].values[::-1][1:]
        t_values = T.T['RE'].values[::-1][1:]
    data = collapse_session_scores(data)
    nr_feat = len(alg1)
    if len(alg1) == 1:
        data = data[data.pipeline.isin([alg1, alg2])]
    data = data.pivot_table(values='score', columns='pipeline',
                            index=['subject', 'dataset'])
    data = data.reset_index(level=['dataset'])
    if fig_ax is None:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
    else:
        ax = fig_ax

    # data.plot.scatter(alg1, alg2, ax=ax)
    # sea.set(rc={'figure.figsize': (nr_feat * 4, 0.1)}, font="Times New Roman")
    fig_ = sea.pairplot(data, x_vars=alg1, y_vars=alg2, hue='dataset', size=6,
                        palette=sea.color_palette("deep", 8),
                        plot_kws={"s": 180},
                        markers=["v", "^", "<", ">", "o", "+", "s", "D"])

    for ind in range(nr_feat):
        ax = fig_.axes[0, ind]
        a_xl = ax.get_xlabel()
        a_yl = ax.get_ylabel()

        ax.set_xlabel(a_xl, fontsize=35)
        ax.set_ylabel(a_yl, fontsize=35)
        ax.set_xlim([0.5, 1])
        ax.set_ylim([0.5, 1])
        ax.set_xticklabels(labels=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=30)
        ax.set_yticklabels(labels=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=30)
        ax.plot([0, 1], [0, 1], ls='--', c='k')
        if ind == 3:
            ax.set_facecolor((0.9176470588235294, 0.9176470588235294,
                              0.9490196078431372, 1.0))
        txt = 't={:.2f}\np={:1.0e}'.format(t_values[ind], p_values[ind])

        if p_values[ind] < 0.05 and t_values[ind] >= 0:
            ax.text(0.51, 0.9, txt, fontsize=35, fontweight='black', color='green')
        elif p_values[ind] < 0.05 and t_values[ind] < 0:
            ax.text(0.51, 0.9, txt, fontsize=35, fontweight='black', color='red')
        else:
            ax.text(0.51, 0.9, txt, fontsize=35)
    mpl.rc('font', family='serif', serif='Times New Roman')
    handles = fig_._legend_data.values()
    labels = fig_._legend_data.keys()
    fig_.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4,
                    fontsize=25)
    fig_.fig.subplots_adjust(top=0.8, bottom=0.08)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    # fig_.savefig('Figure/color_pair_' + task + '.png')
    fig_.savefig('Figure/color_pair_' + task + '.pdf')

    if fig_ax is None:
        return fig
    else:
        return ax


def summary_plot(sig_df, effect_df, sig_pct_df, p_threshold=0.05,
                 title="Features comparison", fig_ax=None):
    '''Visualize significances as a heatmap with green/grey/red for significantly
    higher/significantly lower.
    sig_df is a DataFrame of pipeline x pipeline where each value is a p-value,
    effect_df is a DF where each value is an effect size

    '''
    # effect_df.columns = effect_df.columns.map(_simplify_names)
    # sig_df.columns = sig_df.columns.map(_simplify_names)
    annot_df = effect_df.copy()
    """
    for j, col in enumerate(annot_df.columns):
        for i, row in enumerate(annot_df.index):
            if j <= i:
                '''
                if row == col:
                    effect_df.loc[row, col] = 0
                    sig_df.loc[row, col] = 0
                '''
                txt = '{:.2f}\np={:1.0e}'.format(effect_df.loc[row, col],
                                                 sig_df.loc[row, col])
                if effect_df.loc[row, col] < 0:
                    # we need the effect direction and p-value to coincide.
                    # TODO: current is hack
                    if sig_df.loc[row, col] < p_threshold:
                        sig_df.loc[row, col] = 1e-110
            else:
                txt = ''
            annot_df.loc[row, col] = txt
    """
    for row in annot_df.index:
        for col in annot_df.columns:

            txt = 't={:.2f}\np={:1.0e} ({:.1f}%)'.format(
                effect_df.loc[row, col], sig_df.loc[row, col],
                sig_pct_df.loc[row, col] * 100)
            if effect_df.loc[row, col] > 0:
                pass
            else:
                # we need the effect direction and p-value to coincide.
                # TODO: current is hack
                if sig_df.loc[row, col] < p_threshold:
                    sig_df.loc[row, col] = 1e-110
            annot_df.loc[row, col] = txt

    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig_ax
    """
    mask = np.zeros_like(annot_df)
    mask[np.triu_indices_from(mask)] = True
    """
    palette = sea.light_palette("green", as_cmap=True)
    palette.set_under(color=[1, 1, 1])  # color=[0.8, 0.8, 0.8]
    palette.set_over(color=[0.75, 0, 0])
    sea.heatmap(data=-np.log(sig_df), annot=annot_df,
                fmt='', cmap=palette, linewidths=1, linecolor='0.8',
                annot_kws={'size': 20}, cbar=False, ax=ax,
                vmin=-np.log(0.05), vmax=-np.log(1e-100))
    x_label_text = []
    for tick_label in ax.get_xticklabels():
        tmp_text = tick_label.get_text()  # [-7:] for decomp&feat
        if len(tmp_text) > 20:
            tmp_text = tmp_text[:5] + '_' + tmp_text[-3:]
        else:
            tmp_text = tmp_text
        tick_label.set_rotation(0)
        x_label_text.append(tmp_text)
    ax.set_xticklabels(x_label_text)

    y_label_text = []
    for tick_label in ax.get_yticklabels():
        tmp_text = tick_label.get_text()
        if len(tmp_text) > 10:
            tmp_text = tmp_text[:5] + tmp_text[-3:]
        tick_label.set_rotation(45)
        y_label_text.append(tmp_text)
    ax.set_yticklabels(x_label_text)

    ax.tick_params(axis='y', labelsize=20)  # , rotation=0.9
    ax.tick_params(axis='x', labelsize=20)  # , rotation=0.9
    ax.set_title(title, fontsize=25)
    plt.tight_layout()
    if fig_ax is None:
        return fig
    else:
        return ax


def meta_analysis_plot(stats_df, alg1, alg2, fig_ax=None, pval_fig_ax=None):
    '''A meta-analysis style plot that shows the standardized effect with
    confidence intervals over all datasets for two algorithms.
    Hypothesis is that alg1 is larger than alg2'''
    def _marker(pval):
        if pval < 0.001:
            return '$***$', 100
        elif pval < 0.01:
            return '$**$', 70
        elif pval < 0.05:
            return '$*$', 30
        else:
            raise ValueError('insignificant pval {}'.format(pval))
    assert (alg1 in stats_df.pipe1.unique())
    assert (alg2 in stats_df.pipe1.unique())
    df_fw = stats_df.loc[(stats_df.pipe1 == alg1) & (stats_df.pipe2 == alg2)]
    df_fw = df_fw.sort_values(by='pipe1')
    df_bk = stats_df.loc[(stats_df.pipe1 == alg2) & (stats_df.pipe2 == alg1)]
    df_bk = df_bk.sort_values(by='pipe1')
    dsets = df_fw.dataset.unique()
    ci = []
    mpl.rc('font', family='serif', serif='Times New Roman')
    if fig_ax is None and pval_fig_ax is None:
        order_axes = 0
        nr_axes = 1
        fig = plt.figure()
        gs = gridspec.GridSpec(nr_axes, 5)
        ax = fig.add_subplot(gs[order_axes, :-1])
        pval_ax = fig.add_subplot(gs[order_axes, -1], sharey=ax)
    elif fig_ax is not None and pval_fig_ax is not None:
        ax = fig_ax
        pval_ax = pval_fig_ax

    sig_ind = []
    pvals = []
    ax.set_yticks(np.arange(len(dsets) + 1))
    ori_name = dsets[3]
    # dsets[3] = 'MunichMI'

    y_label_list = ['Meta-effect'] + [_simplify_names(d) for d in dsets]
    # y_label_list[1] = y_label_list[1] + '$^{*}$'
    # y_label_list[-1] = y_label_list[-1] + '$^{*}$'
    # y_label_list[-2] = y_label_list[-2] + '$^{*}$'
    # y_label_list[-3] = y_label_list[-3] + '$^{*}$'
    ax.set_yticklabels(y_label_list,
                       fontsize=14)
    dsets[3] = ori_name
    plt.setp(pval_ax.get_yticklabels(), visible=False)
    _min = 0
    _max = 0
    for ind, d in enumerate(dsets):
        nsub = float(df_fw.loc[df_fw.dataset == d, 'nsub'])
        t_dof = nsub - 1
        ci.append(t.ppf(0.95, t_dof) / np.sqrt(nsub))
        v = float(df_fw.loc[df_fw.dataset == d, 'smd'])
        if v > 0:
            p = df_fw.loc[df_fw.dataset == d, 'p'].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        else:
            p = df_bk.loc[df_bk.dataset == d, 'p'].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        _min = _min if (_min < (v - ci[-1])) else (v - ci[-1])
        _max = _max if (_max > (v + ci[-1])) else (v + ci[-1])
        ax.plot(np.array([v - ci[-1], v + ci[-1]]),
                np.ones((2,)) * (ind + 1), c='tab:grey')
    _range = max(abs(_min), abs(_max))

    ax.set_xlim((0 - _range, 0 + _range))

    ax.set_xticklabels(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=12)
    final_effect = combine_effects(df_fw['smd'], df_fw['nsub'])
    ax.scatter(pd.concat([pd.Series([final_effect]), df_fw['smd']]),
               np.arange(len(dsets) + 1),
               s=np.array([50] + [30] * len(dsets)),
               marker='D',
               c=['k'] + ['tab:grey'] * len(dsets))
    for i, p in zip(sig_ind, pvals):
        m, s = _marker(p)
        ax.scatter(df_fw['smd'].iloc[i],
                   i + 1.4, s=s,
                   marker=m, color='r')
    # pvalues axis stuf
 
    ft_sz = 14
    pval_ax.set_xlim([-0.1, 0.1])
    pval_ax.grid(False)
    pval_ax.set_title('p-value', fontdict={'fontsize': ft_sz + 4})
    pval_ax.set_xticks([])
    for spine in pval_ax.spines.values():
        spine.set_visible(False)
    for ind, p in zip(sig_ind, pvals):
        pval_ax.text(0, ind + 1, horizontalalignment='center',
                     verticalalignment='center',
                     s='{:.2e}'.format(p), fontsize=ft_sz)
    if final_effect > 0:
        p = combine_pvalues(df_fw['p'], df_fw['nsub'])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s,
                       marker=m, c='r')
            pval_ax.text(0, 0, horizontalalignment='center',
                         verticalalignment='center',
                         s='{:.2e}'.format(p), fontsize=ft_sz)
    else:
        p = combine_pvalues(df_bk['p'], df_bk['nsub'])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s,
                       marker=m, c='r')
            pval_ax.text(0, 0, horizontalalignment='center',
                         verticalalignment='center',
                         s='{:.2e}'.format(p), fontsize=ft_sz)

    ax.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, linestyle='--', c='k')
    ax.axhline(0.5, linestyle='-', linewidth=3, c='k')
    # alg1 = 'w/ TS SF'
    # alg_1_title = alg1[:5] + '_' + alg1[-3:] if len(alg1) > 10 else alg1
    # alg_2_title = alg2[:5] + '_' + alg2[-3:] if len(alg2) > 10 else alg2

    alg_1_title = alg1
    alg_2_title = alg2

    nr_blank_up = 30 - 6 - len(alg_2_title)
    nr_blank_down = 45 - 6 - len(alg_1_title) if len(alg_1_title) < 5 else 45 - 14 - len(alg_1_title)

    title = '{}< {} better{}\n{}{} better >'.format(' ' * 0 * 2, alg_2_title,
                                                    ' ' * (nr_blank_up),
                                                    ' ' * (nr_blank_down),
                                                    alg_1_title)
    ax.set_title(title, ha='left', ma='left', loc='left', fontsize=18)
    ax.set_xlabel('Standardized Mean Difference', fontsize=18)

    # mpl.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['ps.fonttype'] = 42

    if fig_ax is None and pval_fig_ax is None:
        fig.tight_layout()
        return fig
    elif fig_ax is not None and pval_fig_ax is not None:
        return ax

# ------------------------ self made function ---------------------------------


def accuracy_plot(nr_row, nr_col, nr_subject, n_ds, acc_all_comp,
                  comp_list, x_y_hue, nr_ses=1, y_scale=None,
                  ci='sd', ylabel='Accuracy', all_subses=False):
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    mpl.rc('font', family='serif', serif='Times New Roman')
    if not all_subses:
        fig_SF = plt.figure(figsize=(10 * nr_col, 4 * nr_row))
        gs = gridspec.GridSpec(nr_row, nr_col)
        axes_SF = []

        for ax_row in range(nr_row):
            for ax_col in range(nr_col):
                axes_SF.append(fig_SF.add_subplot(gs[ax_row, ax_col]))
        for s_id in range(nr_subject):
            for ses_id in range(nr_ses):

                if n_ds == '_Zhou' and s_id == 3 and ses_id > 0:
                    continue

                acc_subj = acc_all_comp.loc[
                    acc_all_comp['subject'] == str(s_id + 1)]
                acc_ses = acc_subj.loc[
                    acc_subj['session'] == str(ses_id + 1)]
                axes_SF[s_id * nr_ses + ses_id].set_title(
                    'subject ' + str(s_id + 1) + ' session ' + str(ses_id + 1),
                    fontsize=20)
                fig_ = sea.pointplot(
                    x=x_y_hue[0], y=x_y_hue[1], hue=x_y_hue[2], data=acc_ses,
                    ci=ci, ax=axes_SF[s_id * nr_ses + ses_id], legend=False,
                    order=comp_list)
                fig_.legend().set_visible(False)
                # Valid when plotting full components (plot one every two)
                fig_.set_xticks(range(0, comp_list[-1], 4))
                fig_.set_xticklabels(
                    [str(i) for i in range(1, comp_list[-1] + 1, 4)],
                    fontsize=16)
                fig_.set_xlabel('Number of applied filters', fontsize=20)
                fig_.set_ylabel(ylabel, fontsize=20)
                if y_scale is not None:
                    fig_.set_yscale(y_scale)
        handles, labels = fig_.get_legend_handles_labels()
        fig_SF.legend(handles=handles, labels=labels, loc='upper center',
                      ncol=3, fontsize=18)
        fig_SF.subplots_adjust(top=0.88, bottom=0.12)

        # fig_SF.suptitle(n_ds[1:], fontsize=18)
        fig_SF.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:

        if len(comp_list) > 64:
            fig_SF = plt.figure(figsize=(20, 5))
        else:
            fig_SF = plt.figure(figsize=(10, 5))

        acc_all_comp['cv'] = acc_all_comp.index
        acc_new = acc_all_comp.drop(
            ['dataset', 'session'], axis=1).groupby(
            ['subject', 'components', 'pipeline'])
        acc_final = acc_new.mean().reset_index().drop('cv', axis=1)

        # fig_ = sea.pointplot(x=x_y_hue[0], y=x_y_hue[1], hue=x_y_hue[2],
        #                      data=acc_final, ci=ci, order=comp_list)

        all_color = sea.color_palette("bright", 10)
        add_color = sea.color_palette("Paired")
        color_manually = [
            all_color[x] for x in [0, 3, 1]]
        # color_manually.append(add_color[3])
        color_manually.append((0.0, 0.0, 0.0))
        # color_manually.append(all_color[2])
        color_manually.append(all_color[3:])
        all_color[5] = (0.0, 0.0, 0.0)

        mpl.rc('font', family='serif', serif='Times New Roman')

        fig_ = sea.tsplot(data=acc_final, time=x_y_hue[0], value=x_y_hue[1],
                          condition=x_y_hue[2], unit='subject',
                          color=all_color)
        fig_.set_xticks(range(0, comp_list[-1], 3))
        fig_.set_xticklabels(
            [str(i) for i in range(1, comp_list[-1] + 1, 3)], fontsize=16)

        y_tick = fig_.get_yticks()
        fig_.set_yticklabels(y_tick, fontsize=16)
        from matplotlib.ticker import FormatStrFormatter

        fig_.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig_.set_xlabel('Number of applied filters', fontsize=24)
        fig_.set_ylabel(ylabel, fontsize=24)
        # fig_.set_title(n_ds[1:], fontsize=24)
        handles, labels = fig_.get_legend_handles_labels()
        # labels[-3] = 'Full TS'
        # labels[-2] = 'TS SF logvar'
        fig_.legend().set_visible(False)
        if len(labels) >= 4:
            if len(comp_list) > 64:
                fig_SF.legend(handles=handles, labels=labels,
                              loc='lower center', ncol=len(labels), fontsize=18)
            else:
                fig_SF.legend(handles=handles, labels=labels,
                              loc='lower center', ncol=int(len(labels) / 2),
                              fontsize=18)
            # np.int(np.ceil(len(labels) / 2))
        else:
            fig_SF.legend(handles=handles, labels=labels,
                          loc='lower center', ncol=3, fontsize=18)
            # np.int(len(labels))
        fig_.xaxis.grid(False)
        fig_SF.subplots_adjust(top=0.85, bottom=0.05)
        # fig_SF.suptitle(n_ds[1:])
        fig_SF.tight_layout(rect=[0, 0.2, 1, 1])

    return fig_SF
