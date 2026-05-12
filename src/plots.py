"""Plot helpers."""

import matplotlib.pyplot as plt
import seaborn as sns


# Yesterday's Sunshine vs Sunshine
def plot_lag_scatter(df):
    # Correlation Calc using the .corr() function
    corr1 = df['sunshine'].corr(df['sunshine_lag_1']).round(3)
    corr2 = df['sunshine'].corr(df['sunshine_lag_2']).round(3)

    f, axes = plt.subplots(1, 2, figsize=(10, 7), sharex=True, sharey=True)
    sns.regplot(data=df,
               x='sunshine_lag_1',
               y='sunshine',
               scatter_kws={'s': 1.0, 'alpha': 0.5},
               ax=axes[0]
               )
    axes[0].set_title('Yesterday\'s Sunshine vs Sunshine')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.6, linestyle='--')
    axes[0].set_xlabel('Yesterday\'s Sunshine (Hrs)')
    axes[0].set_ylabel('Sunshine (Hrs)')
    axes[0].text(14, 7, f"r = {corr1}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    sns.regplot(data=df,
               x='sunshine_lag_2',
               y='sunshine',
               scatter_kws={'s': 1.0, 'alpha': 0.5},
               ax=axes[1]
               )
    axes[1].set_title('Day Before Yesterday\'s Sunshine vs Sunshine')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.6, linestyle='--')
    axes[1].set_xlabel('Day Before Yesterday\'s Sunshine (Hrs)')
    axes[1].set_ylabel('Sunshine (Hrs)')
    axes[1].text(14, 6, f"r = {corr2}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    f.suptitle("Previous Days Sunshine vs Sunshine", fontsize=16, fontstyle="oblique")
    f.tight_layout()
    f.set_facecolor('#D3D3D3')
    return f


# Cloud Cover vs Sunshine by Season
def plot_cloud_cover(df):
    g = sns.FacetGrid(
        df,
        col="Season",
        col_wrap=2,
        height=5,
        aspect=1.2
    )

    g.map_dataframe(sns.boxenplot,
        x = 'cloud_cover',
        y = 'sunshine',
        hue = 'cloud_cover',
        palette = 'pastel',
        legend = False,
        )

    g.set_titles("{col_name}", fontsize=16)
    g.set_axis_labels("Cloud Cover (Hrs)", "Sunshine (Hrs)", fontsize=16)
    g.figure.suptitle("Cloud Cover vs Sunshine by Season", fontsize=18)
    g.figure.tight_layout()
    g.figure.set_facecolor('#D3D3D3')
    return g.figure


# Global Radiation vs Sunshine by Season
def plot_global_radiation(df):
    g = sns.lmplot(data=df,
                x='global_radiation',
                y ='sunshine',
                col='Season',
                hue='Season',
                height=6,
                aspect=0.5,
                palette = 'colorblind',
                # change marker size
                scatter_kws={'s':2}
                )
    g.set_axis_labels("Global Radiation (Wm^-2)", "Sunshine (Hrs)")
    g.figure.suptitle("Global Radiation vs Sunshine by Season")
    g.figure.tight_layout()
    g.figure.set_facecolor('#D3D3D3')
    return g.figure


# Distribution of Pressure by Season
def plot_pressure_histogram(df):
    fig, ax = plt.subplots()
    sns.histplot(data=df,
                 x='pressure',
                 bins=30,
                 kde=True,
                 stat='count',
                 hue='Season',
                 palette='bright',
                 ax=ax)
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of Pressure measurements by Season')
    fig.tight_layout()
    return fig