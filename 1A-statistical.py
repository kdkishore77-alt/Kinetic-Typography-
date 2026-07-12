import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare, chi2_contingency
from docx import Document

def extract_perceptual_data(file_path):
    """
    Extracts frequency data from Tables 6-17 in the RESULT.docx file.
    Maps data to Background Type, Perceptual Dimension, and Spatial Position.
    """
    doc = Document(file_path)
    # Mapping of table indices to their metadata (Background, Dimension)
    # Based on the document structure provided
    table_map = {
        5: ('Radial', 'Aesthetic'), 6: ('Radial', 'Readability'), 7: ('Radial', 'Attention'),
        8: ('Wavy', 'Aesthetic'), 9: ('Wavy', 'Readability'), 10: ('Wavy', 'Attention'),
        11: ('Curly', 'Aesthetic'), 12: ('Curly', 'Readability'), 13: ('Curly', 'Attention'),
        14: ('Flat', 'Aesthetic'), 15: ('Flat', 'Readability'), 16: ('Flat', 'Attention')
    }
    
    extracted_rows = []
    
    for idx, (bg, dim) in table_map.items():
        table = doc.tables[idx]
        # Frequencies are in the second column (index 1), skipping header and 'Total' row
        freqs = []
        for row in table.rows[1:-1]: # Skip header and footer
            freqs.append(int(row.cells[1].text))
        
        extracted_rows.append({
            'Background': bg,
            'Dimension': dim,
            'Frequencies': freqs,
            'Total': sum(freqs)
        })
        
    return pd.DataFrame(extracted_rows)

def perform_statistical_analysis(df):
    """
    Performs Chi-Square Goodness-of-Fit and Independence tests.
    """
    # 1. Goodness of Fit: Is the distribution within one row non-random?
    df['GOF_p_value'] = df['Frequencies'].apply(lambda x: chisquare(x)[1])
    df['is_significant'] = df['GOF_p_value'] < 0.05
    
    # 2. Sensitivity Analysis: Do dimensions differ within each background?
    sensitivity_results = []
    for bg in df['Background'].unique():
        subset = df[df['Background'] == bg]
        # Create contingency table (Dimensions x Positions)
        contingency = np.array(subset['Frequencies'].tolist())
        chi2, p, dof, ex = chi2_contingency(contingency)
        sensitivity_results.append({'Background': bg, 'Sensitivity_p_value': p})
        
    return df, pd.DataFrame(sensitivity_results)

#=========== Bootstrap test ===========

def test_key_redistribution_claim(results_df):
    """
    ONE targeted bootstrap test for the main claim:
    Curly background, Readability dimension: Top-Right vs Center
    """
    # Get the specific data
    row = results_df[
        (results_df['Background'] == 'Curly') & 
        (results_df['Dimension'] == 'Readability')
    ].iloc[0]
    
    freqs = row['Frequencies']  # [TL, BL, C, TR, BR]
    
    # Reconstruct raw participant choices (208 participants)
    choices = []
    for pos_idx, count in enumerate(freqs):
        choices.extend([pos_idx] * count)
    choices = np.array(choices)
    
    # Bootstrap the difference: TR(3) vs Center(2)
    n_bootstrap = 1000
    n = len(choices)
    differences = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample = np.random.choice(choices, size=n, replace=True)
        p_center = np.sum(sample == 2) / n
        p_tr = np.sum(sample == 3) / n
        differences.append(p_tr - p_center)
    
    # Calculate results
    obs_diff = (freqs[3] - freqs[2]) / n * 100
    ci_lower = np.percentile(differences, 2.5) * 100
    ci_upper = np.percentile(differences, 97.5) * 100
    p_value = np.mean(np.array(differences) <= 0) * 2  # Two-tailed
    
    print("\n" + "="*60)
    print("KEY FINDING VALIDATION: Curly Background, Readability Dimension")
    print("="*60)
    print(f"Center selection: {freqs[2]/n*100:.1f}%")
    print(f"Top-Right selection: {freqs[3]/n*100:.1f}%")
    print(f"Difference (TR - Center): {obs_diff:.1f}%")
    print(f"95% Bootstrap CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
    print(f"P-value (bootstrap): {p_value:.4f}")
    print(f"Statistically significant: {ci_lower > 0 or ci_upper < 0}")
    
    return obs_diff, (ci_lower, ci_upper), p_value


#=========== Trade-off test ===========

def test_tradeoff_shift(results_df):
    """
    Test whether the attention-readability correlation changes 
    from Flat to Curly backgrounds
    """
    from scipy.stats import pearsonr
    
    backgrounds = ['Flat', 'Curly']
    correlations = {}
    
    for bg in backgrounds:
        # Get percentages
        att_row = results_df[
            (results_df['Background'] == bg) & 
            (results_df['Dimension'] == 'Attention')
        ].iloc[0]
        
        read_row = results_df[
            (results_df['Background'] == bg) & 
            (results_df['Dimension'] == 'Readability')
        ].iloc[0]
        
        att_pct = np.array(att_row['Frequencies']) / att_row['Total'] * 100
        read_pct = np.array(read_row['Frequencies']) / read_row['Total'] * 100
        
        # Correlation
        r, p = pearsonr(att_pct, read_pct)
        correlations[bg] = {'r': r, 'p': p}
    
    print("\n" + "="*60)
    print("TRADE-OFF SHIFT: Attention-Readability Correlation")
    print("="*60)
    print(f"Flat background: r = {correlations['Flat']['r']:.2f} (p={correlations['Flat']['p']:.3f})")
    print(f"Curly background: r = {correlations['Curly']['r']:.2f} (p={correlations['Curly']['p']:.3f})")
    print(f"Shift: {correlations['Curly']['r'] - correlations['Flat']['r']:.2f}")
    
    # Fisher z-test for difference between correlations
    # (only 5 points, so this is suggestive not definitive)
    return correlations


# ============= Entropy Test ==============

def test_entropy_stability(results_df):
    """
    Compare aesthetic stability vs readability variation
    """
    def entropy(freqs):
        p = np.array(freqs) / np.sum(freqs)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    backgrounds = ['Flat', 'Radial', 'Wavy', 'Curly']
    
    aesthetic_entropies = []
    readability_entropies = []
    
    for bg in backgrounds:
        # Aesthetic
        aest_row = results_df[
            (results_df['Background'] == bg) & 
            (results_df['Dimension'] == 'Aesthetic')
        ].iloc[0]
        aesthetic_entropies.append(entropy(aest_row['Frequencies']))
        
        # Readability
        read_row = results_df[
            (results_df['Background'] == bg) & 
            (results_df['Dimension'] == 'Readability')
        ].iloc[0]
        readability_entropies.append(entropy(read_row['Frequencies']))
    
    # Calculate coefficients of variation
    aest_cv = np.std(aesthetic_entropies) / np.mean(aesthetic_entropies)
    read_cv = np.std(readability_entropies) / np.mean(readability_entropies)
    
    print("\n" + "="*60)
    print("ENTROPY STABILITY: Aesthetic vs Readability")
    print("="*60)
    print(f"Aesthetic entropy CV: {aest_cv:.3f}")
    print(f"Readability entropy CV: {read_cv:.3f}")
    print(f"Ratio (Read/Aest): {read_cv/aest_cv:.2f}x more variable")
    
    return aest_cv, read_cv





#=========== Data-driven plots ===========

def generate_publication_plots(df, sensitivity_df):
    """
    Produces high-resolution, publication-ready visualizations.
    """
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    
    positions = ['1 (TL)', '2 (BL)', '3 (Center)', '4 (TR)', '5 (BR)']
    x = np.arange(len(positions))
    width = 0.25
    colors = {'Attention': '#d62728', 'Readability': '#1f77b4', 'Aesthetic': '#2ca02c'}
    
    backgrounds = ['Flat', 'Radial', 'Wavy', 'Curly'] # Ordered by complexity logic
    
    for i, bg in enumerate(backgrounds):
        ax = axes[i]
        bg_data = df[df['Background'] == bg]
        
        for j, dim in enumerate(['Attention', 'Readability', 'Aesthetic']):
            # Calculate percentages for the Y-axis
            counts = bg_data[bg_data['Dimension'] == dim]['Frequencies'].values[0]
            total = bg_data[bg_data['Dimension'] == dim]['Total'].values[0]
            percentages = [c / total * 100 for c in counts]
            
            ax.bar(x + (j - 1) * width, percentages, width, label=dim, 
                   color=colors[dim], edgecolor='black', linewidth=0.5, alpha=0.85)
        
        # Formatting
        p_sens = sensitivity_df[sensitivity_df['Background'] == bg]['Sensitivity_p_value'].values[0]
        ax.set_title(f'Background: {bg}\n(Inter-Dimension Sensitivity p = {p_sens:.4f})', 
                     fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.set_ylabel('Selection Intensity (%)')
        ax.set_ylim(0, 70)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        if i == 0: ax.legend(frameon=True, loc='upper left')

    plt.tight_layout()
    plt.savefig('1A-perceptual.png', bbox_inches='tight')
    print("Plot saved as '1A-perceptual.png'")


def plot_difference_from_center(results_df):
    import matplotlib.pyplot as plt
    import numpy as np

    positions = ['1 (TL)', '2 (BL)', '3 (Center)', '4 (TR)', '5 (BR)']
    backgrounds = ['Flat', 'Radial', 'Wavy', 'Curly']
    dimensions = ['Attention', 'Readability', 'Aesthetic']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=True)

    for ax, dim in zip(axes, dimensions):
        for bg in backgrounds:
            row = results_df[
                (results_df['Background'] == bg) &
                (results_df['Dimension'] == dim)
            ].iloc[0]

            freqs = np.array(row['Frequencies'])
            percentages = freqs / freqs.sum() * 100
            center_val = percentages[2]

            delta = percentages - center_val
            delta = np.delete(delta, 2)  # remove Center

            ax.plot(
                [p for i, p in enumerate(positions) if i != 2],
                delta, lw=3,
                marker='o', 
                label=bg
            )

        ax.axhline(0, linestyle='--', linewidth=0.8)
        ax.set_title(dim, fontweight='bold')
        ax.set_ylabel(f'$\Delta$ Selection Intensity (%)',fontsize= 20)
        ax.set_xlabel('Spatial Position',fontsize= 20)

    axes[0].legend(title='Background')
    plt.tight_layout()
    plt.savefig('Redistribution.png', bbox_inches='tight')
    plt.show()


def plot_attention_readability_tradeoff(results_df):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    backgrounds = ['Flat', 'Radial', 'Wavy', 'Curly']
    labels = ['TL', 'BL', 'Center', 'TR', 'BR']

    for bg in backgrounds:
        att = results_df[
            (results_df['Background'] == bg) &
            (results_df['Dimension'] == 'Attention')
        ].iloc[0]['Frequencies']

        read = results_df[
            (results_df['Background'] == bg) &
            (results_df['Dimension'] == 'Readability')
        ].iloc[0]['Frequencies']

        att = np.array(att) / sum(att) * 100
        read = np.array(read) / sum(read) * 100

        ax.scatter(att, read, s=80, label=bg)

        for i, lab in enumerate(labels):
            ax.text(att[i] + 0.4, read[i] + 0.4, lab, fontsize=8)

    ax.set_xlabel('Attention Selection (%)',fontsize= 20)
    ax.set_ylabel('Readability Selection (%)',fontsize= 20)
    ax.set_title('Attention–Readability Trade-off Across Backgrounds',
                 fontweight='bold')
    ax.legend(title='Background')
    plt.tight_layout()
    plt.savefig('Attention-Readability.png', bbox_inches='tight')
    plt.show()


def plot_perceptual_dispersion(results_df):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def entropy(freqs):
        p = np.array(freqs) / np.sum(freqs)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    backgrounds = ['Flat', 'Radial', 'Wavy', 'Curly']
    dimensions = ['Attention', 'Readability', 'Aesthetic']

    records = []

    for dim in dimensions:
        for bg in backgrounds:
            freqs = results_df[
                (results_df['Background'] == bg) &
                (results_df['Dimension'] == dim)
            ].iloc[0]['Frequencies']

            records.append({
                'Dimension': dim,
                'Background': bg,
                'Entropy': entropy(freqs)
            })

    ent_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for dim in dimensions:
        sub = ent_df[ent_df['Dimension'] == dim]
        ax.plot(sub['Background'], sub['Entropy'],
                marker='o', label=dim)

    ax.set_ylabel('Perceptual Dispersion (Entropy)',fontsize= 20)
    ax.set_xlabel('Background Complexity',fontsize= 20)
    ax.set_title('Perceptual Dispersion Across Backgrounds',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Perceptual-Dispersion.png', bbox_inches='tight')
    plt.show()


    
# --- Execution Flow ---
if __name__ == "__main__":
    # 1. Data Extraction
    results_df = extract_perceptual_data('RESULT.docx')
    
    # 2. Chi-square analysis 
    results_df, sensitivity_df = perform_statistical_analysis(results_df)
    
    # 3. Visualization
    generate_publication_plots(results_df, sensitivity_df)
    plot_difference_from_center(results_df)
    plot_attention_readability_tradeoff(results_df)
    plot_perceptual_dispersion(results_df)


    # 4. Three targeted bootstrap tests
    print("\n" + "="*60)
    print("TARGETED STATISTICAL VALIDATION")
    print("="*60)
    
    diff, ci, p = test_key_redistribution_claim(results_df)
    correlations = test_tradeoff_shift(results_df)
    aest_cv, read_cv = test_entropy_stability(results_df)
    
    # 5. Save summary for manuscript
    with open('statistical_summary_for_paper.txt', 'w') as f:
        f.write("KEY STATISTICAL FINDINGS\n")
        f.write("=======================\n\n")
        f.write(f"1. Redistribution Effect (Curly/Readability):\n")
        f.write(f"   TR vs Center difference: {diff:.1f}% (95%% CI [{ci[0]:.1f}, {ci[1]:.1f}], p={p:.4f})\n\n")
        f.write(f"2. Trade-off Shift:\n")
        f.write(f"   Flat correlation: r={correlations['Flat']['r']:.2f}\n")
        f.write(f"   Curly correlation: r={correlations['Curly']['r']:.2f}\n\n")
        f.write(f"3. Entropy Stability:\n")
        f.write(f"   Aesthetic CV: {aest_cv:.3f}\n")
        f.write(f"   Readability CV: {read_cv:.3f}\n")
