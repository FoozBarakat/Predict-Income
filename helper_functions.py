import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, precision_score, recall_score, accuracy_score

from sklearn.metrics import roc_curve, auc, precision_recall_curve

sns.set_style("darkgrid")
sns.set_palette("muted")
heatmap_cmap= "Blues"


""" Ploting Functions
"""

def explore_categorical(df, x, hue, fillna=True, placeholder= 'MISSING', figsize=(6,4), order=None):
    # Make a copy of the dataframe and fillna
    temp_df= df.copy()

    # Before filling nulls, save null value counts and percent for printing
    null_count= temp_df[x].isna().sum()
    null_perc= null_count/len(temp_df)* 100

    # fillna with placeholder
    if fillna == True:
        temp_df[x] = temp_df[x].fillna(placeholder)
    
    # Create figure with desired figsize
    fig, ax = plt.subplots(figsize=figsize)

    # Plotting a count plot
    sns.countplot(data= temp_df, x= x, ax= ax, order= order, hue= hue)
  
    # Rotate Tick Labels for long names
    ax.set_xticklabels(ax.get_xticklabels(), rotation= 45, ha= 'right')
  
    # Add a title with the feature name included
    ax.set_title(f"Column: {x}")

    # Add labels to the bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=10);

    # Fix layout and show plot (before print statements)
    fig.tight_layout()
    plt.show()

    # Print null value info
    print(f"- NaN's Found: {null_count} ({round(null_perc,2)}%)")
    
    # Print cardinality info
    nunique = temp_df[x].nunique()
    print(f"- Unique Values: {nunique}")
  
    # First find value counts of feature
    val_counts = temp_df[x].value_counts(dropna=False)
  
    # Define the most common value
    most_common_val = val_counts.index[0]
    # Define the frequency of the most common value
    freq = val_counts.values[0]
  
    # Calculate the percentage of the most common value
    perc_most_common = freq / len(temp_df) * 100
  
    # Print the results
    print(f"- Most common value: '{most_common_val}' occurs {freq} times ({round(perc_most_common,2)}%)")
    
    # print message if quasi-constant or constant (most common val more than 98% of data)
    if perc_most_common > 98:
        print(f"\n- [!] Warning: '{x}' is a constant or quasi-constant feature and should be dropped.")
    else:
        print("- Not constant or quasi-constant.")

    return fig, ax;

def explore_numeric(df, x, figsize= (8,5)):
    # calculate the mean
    mean= df[x].mean().round(2)
    # calculate the median
    median= df[x].median().round(2)

    # Making our figure with gridspec for subplots
    gridspec = {'height_ratios':[0.7,0.3]}
    fig, axes = plt.subplots(nrows=2, figsize=figsize, sharex=True, gridspec_kw=gridspec)

    # Histogram on Top
    sns.histplot(data=df, x=x, ax=axes[0])
    # Boxplot on Bottom
    sns.boxplot(data=df, x=x, ax=axes[1])

    axes[0].axvline(mean, color= 'red', label= f'Mean is {mean}', lw= 2)
    axes[0].axvline(median, color= 'orange', label= f'Median is {median}', lw= 2, ls= ':')

    # Adding a title and kegend
    axes[0].set_title(f"Column: {x}")
    axes[0].legend(loc= (1.01, .75))
  
    # Adjusting subplots to best fill Figure
    fig.tight_layout()
    # Ensure plot is shown before message
    plt.show()

    # Print message with info on the count and % of null values
    null_count = df[x].isna().sum()
    null_perc = null_count/len(df)* 100
    print(f"- NaN's Found: {null_count} ({round(null_perc,2)}%)")
   
   # First find value counts of feature
    val_counts = df[x].value_counts(dropna=False)
  
    # Define the most common value
    most_common_val = val_counts.index[0]
  
    # Define the frequency of the most common value
    freq = val_counts.values[0]
  
    # Calculate the percentage of the most common value
    perc_most_common = freq / len(df) * 100
  
    # Print the results
    print(f"- Most common value: '{most_common_val}' occurs {freq} times ({round(perc_most_common,2)}%)")
  
    # print message if quasi-constant or constant (most common val more than 98% of data)
    if perc_most_common > 98:
        print(f"\n- [!] Warning: '{x}' is a constant or quasi-constant feature and should be dropped.")
    else:
        print("- Not constant or quasi-constant.")
    return fig, axes;

def plot_categorical_vs_target(df, x, y, figsize=(8, 4), fillna=True, placeholder='MISSING', order=None, target_type='reg'):
    # Make a copy of the dataframe and fillna
    temp_df = df.copy()

    # Fill NaN values with placeholder
    if fillna:
        temp_df[x] = temp_df[x].fillna(placeholder)
    else:  # Drop rows with nulls in column x
        temp_df = temp_df.dropna(subset=[x])

    # If order is specified, convert column to a categorical dtype
    if order is not None:
        temp_df[x] = pd.Categorical(temp_df[x], categories=order, ordered=True)

    # Create the figure and subplots
    fig, ax = plt.subplots(figsize=figsize)

    # REGRESSION-TARGET PLOT
    if target_type == 'reg':
        # Barplot
        sns.barplot(data=temp_df, x=x, y=y, ax=ax, order=order, alpha=0.6, linewidth=1, edgecolor='black', errorbar=None)
        # Stripplot
        sns.stripplot(data=temp_df, x=x, y=y, hue=x, ax=ax, order=order, hue_order=order, legend=False, edgecolor='white', linewidth=0.5, size=5, zorder=1)

    # CLASSIFICATION-TARGET PLOT
    elif target_type == 'class':
        sns.histplot(
            data=temp_df,
            hue=y,
            x=x,
            stat='percent',
            multiple='fill',
            ax=ax,
            hue_order=temp_df[y].unique(),  # Respect the hue order
            element='bars'
        )

    # Rotate xlabels
    ax.set_xticks(ax.get_xticks())  # Avoid a bug in some matplotlib versions
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Add a title
    ax.set_title(f"{x} vs. {y}")
    fig.tight_layout()

    return fig, ax


def plot_numeric_vs_target(df, x, y, figsize=(6,4), target_type='reg', estimator='mean', errorbar='ci', sorted=False ,ascending=False, **kwargs): 
    nulls = df[[x,y]].isna().sum()
    if nulls.sum()>0:
        print(f"- Excluding {nulls.sum()} NaN's")
        temp_df = df.dropna(subset=[x,y,])
    else:
        temp_df = df
  
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # REGRESSION-TARGET PLOT
    if 'reg' in target_type:
        # Calculate the correlation
        corr = df[[x,y]].corr().round(2)
        r = corr.loc[x,y]
    
        # Plot the data
        scatter_kws={'edgecolors':'white','linewidths':1,'alpha':0.8}
        sns.regplot(data=temp_df, x=x, y=y, ax=ax, scatter_kws=scatter_kws, **kwargs) # Included the new argument within the sns.regplot function
    
        ## Add the title with the correlation
        ax.set_title(f"{x} vs. {y} (r = {r})")

    # CLASSIFICATION-TARGET PLOT
    elif 'class' in target_type:
        # Sort the groups by median/mean
        if sorted == True:
            if estimator == 'median':
                group_vals = temp_df.groupby(y)[x].median()
            elif estimator =='mean':
                group_vals = temp_df.groupby(y)[x].mean()
            
            ## Sort values
            group_vals = group_vals.sort_values(ascending=ascending)
            order = group_vals.index
        else:
            # Set order to None if not calcualted
            order = None

        # Left Subplot (barplot)
        sns.barplot(data=temp_df, x=y, y=x, order=order,  estimator=estimator, errorbar=errorbar, ax=ax, **kwargs)

        # Add title
        ax.set_title(f"{x} vs. {y}")

    # rotate xaxis labels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


    # Final Adjustments & return
    fig.tight_layout()
    fig.show()

    return fig, ax;


""" Evaluation Functions
"""


def regression_metrics(y_true, y_pred, label=''):
    # Get metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)


    metrics = {'Label':label, 'MAE':mae, 'MSE':mse, 'RMSE':rmse, 'R^2':r_squared}

    return metrics


def evaluate_regression(reg, X_train, y_train, X_test, y_test):
    # Get predictions for training data
    y_train_pred= reg.predict(X_train)

    # Call the helper function to obtain regression metrics for training data
    results_train= regression_metrics(y_train, y_train_pred, label='Training Data')

    # Get predictions for test data
    y_test_pred= reg.predict(X_test)
    # Call the helper function to obtain regression metrics for test data
    results_test= regression_metrics(y_test, y_test_pred, label='Test Data' )

    # Store results in a dataframe if ouput_frame is True
    results_df= pd.DataFrame([results_train,results_test])
    # Set the label as the index
    results_df= results_df.set_index('Label')
    # Set index.name to none to get a cleaner looking result
    results_df.index.name= None

    return results_df.round(3)

def classification_metrics(y_true, y_pred, label='', output_dict=False, figsize=(8,4), normalize='true', cmap='Blues', colorbar=False):
    # Get the classification report
    report = classification_report(y_true, y_pred)
  
    header= "-"*70
    print(header, f" Classification Metrics: {label}", header, sep='\n')
    print(report)
  
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    # create a confusion matrix  of raw counts
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize=None, cmap='gist_gray', colorbar=colorbar, ax = axes[0],);
    axes[0].set_title("Raw Counts")
  
    # create a confusion matrix with the test data
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize=normalize, cmap=cmap, colorbar=colorbar, ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")
  
    # Adjust layout and show figure
    fig.tight_layout()
    plt.show()
  
    # Return dictionary of classification_report
    if output_dict==True:
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        return report_dict
    
def evaluate_classification(model, X_train, y_train, X_test, y_test, figsize=(6,4), normalize='true', output_dict = False, cmap_train=heatmap_cmap, cmap_test="Reds",colorbar=False):
    # Get predictions for training data
    y_train_pred = model.predict(X_train)
  
    # Call the helper function to obtain regression metrics for training data
    results_train = classification_metrics(y_train, y_train_pred, output_dict=True, figsize=figsize, colorbar=colorbar, cmap=cmap_train, label='Training Data')
    print()
  
    # Get predictions for test data
    y_test_pred = model.predict(X_test)
    # Call the helper function to obtain regression metrics for test data
    results_test = classification_metrics(y_test, y_test_pred, output_dict=True,figsize=figsize, colorbar=colorbar, cmap=cmap_test, label='Test Data' )
    
    if output_dict == True:
        # Store results in a dataframe if ouput_frame is True
        results_dict = {'train':results_train, 'test': results_test}
        return results_dict



""" Coefficients and Importances Functions
"""


def annotate_hbars(ax, ha='left', va='center', size=12, xytext=(4,0), textcoords='offset points'):
    for bar in ax.patches:
        ## calculate center of bar
        bar_ax = bar.get_y() + bar.get_height() / 2
        ## get the value to annotate
        val = bar.get_width()
        if val < 0:
            val_pos = 0
        else:
            val_pos = val
        # ha and va stand for the horizontal and vertical alignment
        ax.annotate(f"{val:.3f}", (val_pos,bar_ax), ha=ha, va=va, size=size, xytext=xytext, textcoords=textcoords)

def plot_coeffs(coeffs, top_n=None, figsize=(4,5), intercept=False, intercept_name="intercept", annotate=False, ha='left', va='center', size=12, xytext=(4,0), textcoords='offset points'):

    """ Plots the top_n coefficients from a Series, with optional annotations.
    """
    # Drop intercept if intercept=False and
    if (intercept == False) & (intercept_name in coeffs.index):
        coeffs = coeffs.drop(intercept_name)
    
    if top_n == None:
        ## sort all features and set title
        plot_vals = coeffs.sort_values()
        title = "All Coefficients - Ranked by Magnitude"
    else:
        ## rank the coeffs and select the top_n
        coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
        top_n_features = coeff_rank.head(top_n)

        ## sort features and keep top_n and set title
        plot_vals = coeffs.loc[top_n_features.index].sort_values()
        title = f"Top {top_n} Largest Coefficients"

    ## plotting top N importances
    ax = plot_vals.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Coefficient', ylabel='Feature Names', title=title)
    ax.axvline(0, color='k')

    if annotate == True:
        annotate_hbars(ax, ha=ha, va=va, size=size, xytext=xytext, textcoords=textcoords)

    return ax;

def plot_importance(importances, top_n=None,  figsize=(8,6)):

    # sorting with asc=false for correct order of bars
    if top_n==None:
        ## sort all features and set title
        plot_vals = importances.sort_values()
        title = "All Features - Ranked by Importance"
    else:
        ## sort features and keep top_n and set title
        plot_vals = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"
   
    
    ax = plot_vals.plot(kind='barh', figsize=figsize)

    # set titles and axis labels
    ax.set(xlabel='Importance', ylabel='Feature Names', title=title)

    return ax;


def tuning_threshold(model, X_test, y_test):
    # Get the model probability predictions for the test set
    test_probs = model.predict_proba(X_test)

    # Create an array of float values between 0 and 1 with a step size of .05
    thresholds = np.arange(start=0, stop=1.05, step=.05)

    # Create empty recall and precision lists
    recalls = []
    precisions = []
    accuracies = []

    # Iterate over thresholds
    for thresh in thresholds:
        # Convert probabilities to predictions according to each threshold
        preds = convert_probs_to_preds(test_probs, thresh)
        # Record the recall and precision for predictions at that threshold
        recalls.append(recall_score(y_test, preds))
        precisions.append(precision_score(y_test, preds))
        accuracies.append(accuracy_score(y_test, preds))

    # Plot precisions and recalls for each probability
    plt.figure(figsize=(15,5))
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.legend()
    plt.title('Precision, Recall, and Accuracy Scores Across Decision Thresholds')
    plt.xlabel('Decision Thresholds')
    plt.ylabel('Score')
    plt.grid()
    plt.xticks(thresholds)
    plt.show()

    return test_probs



def convert_probs_to_preds(probabilities, threshold, pos_class=1):
    predictions = [1 if prob[pos_class] > threshold else 0 for prob in probabilities]
    return predictions

def evaluate_tunded__threshold(threshold, test_probs, y_test):
    test_preds = convert_probs_to_preds(test_probs, threshold)

    classification_metrics(y_test, test_preds, output_dict=True,figsize=(6,4), colorbar=False, cmap='Greens', label='Test Data' )  

    return

def plot_roc_curve(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        # Get the probability scores for the positive class
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve with AUC in the label
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plot the diagonal line for random chance
    plt.plot([0, 1], [0, 1], '--', label='--', color='black')

    # Customize plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.show()

def plot_recall_vs_threshold(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)  

        # Plot Recall vs. Threshold 
        plt.plot(thresholds, recall[:-1], label=f'{model_name}')

    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs. Threshold')
    plt.legend(loc='best')
    plt.show()
