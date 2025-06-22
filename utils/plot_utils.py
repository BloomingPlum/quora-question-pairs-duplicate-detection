import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_count_distribution(
    df, col1, col2, label_col, 
    count_label="Count", 
    title="Distribution by Duplicate Status",
    colors={0: 'brown', 1: 'yellow'}
):
    """
    Plots a boxplot comparing the distribution of counts (char/word) for two question columns by label.
    Args:
        df: DataFrame containing the columns.
        col1: First question count column (e.g., 'question1_character_count').
        col2: Second question count column (e.g., 'question2_character_count').
        label_col: Column indicating label (e.g., 'is_duplicate').
        count_label: Y-axis label (default "Count").
        title: Plot title.
        colors: Dict for hue values (default brown/yellow).
    """
    plt.figure(figsize=(12, 6))
    
    melted = pd.melt(
        df[[col1, col2, label_col]], 
        id_vars=[label_col],
        value_vars=[col1, col2],
        var_name='question_type',
        value_name=count_label
    )
    # Clean up question_type for axis
    melted['question_type'] = melted['question_type'].map({
        col1: 'Question 1',
        col2: 'Question 2'
    })
    
    ax = sns.boxplot(
        data=melted, 
        x='question_type', y=count_label, 
        hue=label_col, 
        palette=colors, 
        showfliers=False
    )
    
    plt.title(title)
    plt.xlabel('Question Type')
    plt.ylabel(count_label)
    plt.xticks(rotation=0)
    
    # Fix legend: manually map handles to new labels
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Not Duplicate', 'Duplicate'], title='Is Duplicate')
    plt.tight_layout()
    plt.show()


def class_distribution(df, column):
    """
    Plots the class distribution of a specified column in a DataFrame as a bar chart 
    with percentage labels.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the column for which to plot the class distribution.

    Returns:
    --------
    None
        This function displays a matplotlib bar plot and does not return anything.
    """
    value_counts = df[column].value_counts()
    total = value_counts.sum()

    # Plot the bar chart
    ax = value_counts.plot(kind='bar')

    # Add percentage labels
    percent_labels = [f"{(v / total * 100):.1f}%" for v in value_counts]
    ax.bar_label(ax.containers[0], labels=percent_labels)

    # Show the plot
    plt.show()

