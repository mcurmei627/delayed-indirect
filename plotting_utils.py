import numpy as np
from simulation import *
import matplotlib.pyplot as plt

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_context():
    TITLE_SIZE = 25
    LABEL_SIZE = 20
    LEGEND_TITLE_SIZE = 18
    LEGEND_SIZE = 15
    TICK_SIZE = 15
    FONT = 'serif'
    params = {}
    params['legend.title_fontsize'] = LEGEND_TITLE_SIZE
    params['axes.labelsize'] = LABEL_SIZE
    params['axes.titlesize'] = TITLE_SIZE
    params['legend.fontsize'] = LEGEND_SIZE
    params["xtick.labelsize"]= TICK_SIZE
    params["ytick.labelsize"] = TICK_SIZE
    params["font.family"] = "Times New Roman"
    return params, FONT

def plot_conclusion(conclusion_list, metric, ax=None, pretty_name=None, plot_color=None, plot_time=None, plot_range=[20, 400]):
    if ax is None:
        fig, ax = plt.subplots()
    for c, time_range in conclusion_list:
        if plot_time is not None:
            if len(time_range) != 0 and (max(time_range)+1 not in plot_time):
                continue
        x = np.mean(c.avg_metrics['time'], 0)
        if metric == 'mediated_ph2_frac':
            avg_m , ci_m = c.mean_confidence_interval(c.avg_metrics['phase2_mediated'])
            avg_um, ci_um = c.mean_confidence_interval(c.avg_metrics['phase2_unmediated'])
            avg_m = np.array(avg_m)
            avg_um = np.array(avg_um)
            avg = avg_m / (avg_m + avg_um) * 100
            ci_m = np.array(ci_m)
            ci_um = np.array(ci_um)
        elif metric == 'ph2_bifrac':
            avg_m_mono = np.array(c.mean_confidence_interval(c.avg_metrics['phase2_mediated_mono'])[0])
            avg_m_bi = np.array(c.mean_confidence_interval(c.avg_metrics['phase2_mediated_bi'])[0])
            avg_um_mono = np.array(c.mean_confidence_interval(c.avg_metrics['phase2_unmediated_mono'])[0])
            avg_um_bi = np.array(c.mean_confidence_interval(c.avg_metrics['phase2_unmediated_bi'])[0])
            avg_m_bifrac = smooth(avg_m_bi / (avg_m_mono + avg_m_bi) * 100, 2)
            avg_um_bifrac = smooth(avg_um_bi / (avg_um_mono + avg_um_bi) * 100, 2)
        else:
            avg , ci = c.mean_confidence_interval(c.avg_metrics[metric])
            avg = np.array(avg)
            ci = np.array(ci)

        if len(time_range) == 0:
            label = 'Natural growth'
            color = 'black'
            lw = 3
            alpha = 1
        else:
            label = f'Intervention: t={min(time_range)}-{max(time_range)+1}'
            color = 'tab:blue' if plot_color is None else plot_color
            alpha = 0.1 + 0.9 * max(time_range)/plot_range[1]
            lw =2
        
        if metric == 'ph2_bifrac':
            ax.plot(x, avg_m_bifrac, color=color, linestyle='--', alpha=1, lw=lw)
            ax.plot(x, avg_um_bifrac, color=color, alpha=1, lw=lw)
        else:
            ax.plot(x, avg, label=label, color=color, alpha=alpha, lw=lw)
    ax.set_xlim(plot_range)
    ax.set(xlabel='Time', ylabel=metric if pretty_name is None else pretty_name)
    return ax

def plot_indirect(unmed_clist, med_clist, metric, ax=None, pretty_name=None, plot_color=None, plot_time=None, plot_range=[20, 400]):
    for i in range(len(med_clist)):
        c_m, time_range0 = med_clist[i]
        c_um, time_range1 = unmed_clist[i]
        if time_range0 != time_range1:
            return i
        if plot_time is not None:
            if len(time_range0) != 0 and (max(time_range0)+1 not in plot_time):
                continue
            
        x = np.mean(c_m.avg_metrics['time'], 0)

        avg_m = np.array(c_m.mean_confidence_interval(c_m.avg_metrics[metric])[0])
        avg_um = np.array(c_um.mean_confidence_interval(c_um.avg_metrics[metric])[0])
        if len(time_range0) == 0:
            label = 'Natural growth'
            color = 'black'
            lw = 3
            alpha = 1
            ax.plot(x, avg_m, label=label, color=color,alpha=alpha, lw=lw)

        else:
            label = f'Intervention: t={min(time_range0)}-{max(time_range0)+1}'
            color = 'tab:blue' if plot_color is None else plot_color
            alpha = 0.3 + 0.7 * max(time_range0)/plot_range[1]
            lw =2
            ax.plot(x, avg_m, label=label, color=color,alpha=alpha, lw=lw)
            ax.plot(x, avg_um, label=label, color=color, linestyle='--', alpha=alpha, lw=lw)
    
    ax.set_xlim(plot_range)
    ax.set(xlabel='Time', ylabel=metric if pretty_name is None else pretty_name)
    return ax

def plot_diff_frac(unmed_clist, med_clist, metric, nat_idx, ax=None, pretty_name=None, plot_color=None, smooth_pts=None, plot_time=None, plot_range=[20, 400]):
    c_nat = med_clist[nat_idx][0]
    for i in range(len(med_clist)):
        c_m, time_range0 = med_clist[i]
        c_um, time_range1 = unmed_clist[i]
        if time_range0 != time_range1:
            return i
        if len(time_range0) == 0 or (max(time_range0)+1 not in plot_time):
            continue
        
        x = np.mean(c_m.avg_metrics['time'], 0)

        avg_m = np.array(c_m.mean_confidence_interval(c_m.avg_metrics[metric])[0])
        avg_um = np.array(c_um.mean_confidence_interval(c_um.avg_metrics[metric])[0])
        avg_nat = np.array(c_nat.mean_confidence_interval(c_nat.avg_metrics[metric])[0])
        avg = (avg_m - avg_um) / (avg_m - avg_nat)
        if smooth_pts is not None:
            avg = smooth(avg, smooth_pts)
        avg = np.clip(avg, 0, None)
        avg[11:14] = 0
        label = f'Intervention: t={min(time_range0)}-{max(time_range0)+1}'
        color = 'tab:blue' if plot_color is None else plot_color
        alpha = 0.3 + 0.7 * max(time_range0)/plot_range[1]
        lw =2
        ax.plot(x, avg, label=label, color=color,alpha=alpha, lw=lw)
    ax.set_xlim(plot_range)
    ax.set(xlabel='Time', ylabel=metric if pretty_name is None else pretty_name)
    return ax
