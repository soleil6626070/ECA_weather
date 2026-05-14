"""Plot helpers."""

import matplotlib.pyplot as plt
import seaborn as sns


# Yesterday's Sunshine vs Sunshine
def plot_lag_scatter(df):
    # Correlation Calc using the .corr() function
    corr1 = df['sunshine'].corr(df['sunshine_lag_1']).round(3)
    corr2 = df['sunshine'].corr(df['sunshine_lag_2']).round(3)

    f, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    sns.regplot(data=df,
               x='sunshine_lag_1',
               y='sunshine',
               scatter_kws={'s': 1.0, 'alpha': 0.5},
               ax=axes[0]
               )
    axes[0].set_title("Yesterday's Sunshine vs Sunshine", fontsize=16)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.6, linestyle='--')
    axes[0].set_xlabel("Yesterday's Sunshine (Hrs)", fontsize=14)
    axes[0].set_ylabel('Sunshine (Hrs)', fontsize=14)
    axes[0].tick_params(axis='both', labelsize=12)
    axes[0].text(14, 7, f"r = {corr1}", fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

    sns.regplot(data=df,
               x='sunshine_lag_2',
               y='sunshine',
               scatter_kws={'s': 1.0, 'alpha': 0.5},
               ax=axes[1]
               )
    axes[1].set_title("Day Before Yesterday's Sunshine vs Sunshine", fontsize=16)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.6, linestyle='--')
    axes[1].set_xlabel("Day Before Yesterday's Sunshine (Hrs)", fontsize=14)
    axes[1].set_ylabel('Sunshine (Hrs)', fontsize=14)
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].text(14, 6, f"r = {corr2}", fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

    f.suptitle("Previous Days Sunshine vs Sunshine", fontsize=22, fontstyle="oblique")
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

    g.set_titles("{col_name}", size=16)
    g.set_axis_labels("Cloud Cover (Hrs)", "Sunshine (Hrs)", fontsize=16)
    g.figure.suptitle("Cloud Cover vs Sunshine by Season", fontsize=18)
    g.figure.tight_layout()
    g.figure.set_facecolor('#D3D3D3')
    return g.figure


# Global Radiation vs Sunshine by Season
def plot_global_radiation(df):
    seasons = [['Winter', 'Summer'], ['Autumn', 'Spring']]
    palette = dict(zip(['Winter', 'Spring', 'Summer', 'Autumn'], sns.color_palette('colorblind', 4)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [1, 2]})
    for row in range(2):
        for col in range(2):
            season = seasons[row][col]
            ax = axes[row][col]
            sns.regplot(data=df[df['Season'] == season],
                        x='global_radiation',
                        y='sunshine',
                        scatter_kws={'s': 2},
                        color=palette[season],
                        ax=ax)
            ax.set_title(season)
            ax.set_xlabel("Global Radiation (Wm^-2)", fontsize=14)
            ax.set_ylabel("Sunshine (Hrs)", fontsize=14)

    axes[0][0].set_xlim(0, 200)
    axes[1][0].set_xlim(0, 200)
    for ax in axes.flat:
        ax.set_ylim(-0.5, 17.5)

    fig.suptitle("Global Radiation vs Sunshine by Season", fontsize=18)
    fig.tight_layout()
    fig.set_facecolor('#D3D3D3')
    return fig


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