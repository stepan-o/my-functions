import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from collections.abc import Iterable
from gensim.models.doc2vec import LabeledSentence
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from time import time


# -------------------- General plotting ---------------------------------


def plot_time_series(series_to_plot, summary_stats=False,
                     create_plot=True, show_plot=True, ax=None,
                     color='blue', linestyle='-', linewidth=1, alpha=1.,
                     plot_title="", ylabel="", xlabel="Date", tick_label_size=14,
                     x_log=False, y_log=False,
                     highlight_0=True, highlight_0_color='black',
                     minmax=True, mean=True, median=True, units="",
                     highlight=None, caption_lift=1.03):
    """
    a function to plot a line plot of provided time series

    (optional) highlights a period of time with an orange rectangle
    (optional) plots minmax, mean, median of the provided Series
    (optional) x and y axes can (separately) be set to logarithmic scales

    can be used to plot several lines on the same plot
    via several calls to this function
    parameters 'create_plot', 'show_plot', and 'ax' are used to control
    several plots as follows:

    if only one line --'create_plot' = True, 'show_plot' = True
                                        (default)
    if more then one line -- first plot -- create_plot = True, show_plot = False
                       subsequent plots -- create_plot = False, show_plot = False
                                           need to get axis created from the first plot
                                           through ax = plt.gca()
                                           and provide to this function in the second call
                                           as ax = ax
                             final plot    create_plot = False, show_plot = True, ax = ax
    ------------------- plot parameters -----------------------------
    :param series_to_plot: pandas.Series -- Series to be plotted
    :param summary_stats:  boolean       -- whether to show summary stats for the Series
    :param create_plot:    boolean       -- whether to create figure and axis
                                            (set to False for subsequent plots on same axis)
    :param show_plot:      boolean       -- whether to show the plot
                                            (set to False for subsequent plots on same axis)
    :param ax:          matplotlib axis  -- if provided, plot on this axis
                                            (if subsequent plot, provide ax)
    ----------------- main line parameters --------------------------
    :param color:          string        -- color to be used for the main line (matplotlib)
    :param linestyle:      string        -- linestyle to be used for the main line  (matplotlib)
    :param linewidth:      int           -- linewidth to be used for the main line (matplotlib)
    :param alpha:          alpha         -- alpha (transparency) to be used for the main line
    ------------------- axis parameters -----------------------------
    :param plot_title:     string        -- title of the chart
    :param xlabel:         string        -- label for x axis
    :param ylabel:         string        -- label for y axis
    :param tick_label_size:    int       -- size of labels on x and y ticks
    :param y_log:          boolean       -- whether to use log scale for y
    :param x_log:          boolean       -- whether to use log scale for x
    ------------- highlight origin parameters -----------------------
    :param highlight_0:    boolean       -- whether to highlight to origin (y=0)
    :param highlight_0_color:  string    -- color used to highlight the origin
    ----------- min, max, mean, median, units -----------------------
    :param minmax:         boolean       -- whether to plot the min and max values
    :param mean :          boolean       -- whether to plot the mean
    :param median:         boolean       -- whether to plot the median
    :param units:          string        -- units to be added at the end of captions

    :param caption_lift:   float         -- value used to lift captions above lines
    :param highlight:      list          -- list of min x and max x of plot region to highlight
    """
    if summary_stats:
        print(ylabel, "summary statistics")
        print(series_to_plot.describe())

    # set font parameters
    font = dict(family='serif', color='darkred', weight='normal', size=16)

    if create_plot:
        # create figure and axis
        f, ax = plt.subplots(1, figsize=(8, 8))

    # plot the time series
    ax.plot(series_to_plot, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha)

    if y_log:
        # set y scale to logarithmic
        ax.set_yscale('log')

    if x_log:
        # set x scale to logarithmic
        ax.set_xscale('log')

    if highlight_0:
        # draw a horizontal line at 0
        ax.axhline(0, linestyle='--', linewidth=2, color=highlight_0_color)

    if minmax:
        # highlight min and max
        ser_min = series_to_plot.min()
        ax.axhline(ser_min, linestyle=':', color='red')
        ax.text(series_to_plot.index[len(series_to_plot) // 3], ser_min * caption_lift,
                "Min: {0:.2f}{1}".format(ser_min, units), fontsize=14)
        ser_max = series_to_plot.max()
        ax.axhline(series_to_plot.max(), linestyle=':', color='green')
        ax.text(series_to_plot.index[len(series_to_plot) // 3], ser_max * caption_lift,
                "Max: {0:.2f}{1}".format(ser_max, units), fontsize=14)

    if mean:
        # plot Series mean
        ser_mean = series_to_plot.mean()
        ax.axhline(ser_mean, linestyle='--', color='deeppink')
        ax.text(series_to_plot.index[len(series_to_plot) // 3], ser_mean * caption_lift,
                "Mean: {0:.2f}{1}".format(ser_mean, units), fontsize=14)

    if median:
        # plot Series median
        ser_median = series_to_plot.median()
        ax.axhline(ser_median, linestyle=':', color='blue')
        ax.text(series_to_plot.index[int(len(series_to_plot) * 0.7)], ser_median * caption_lift,
                "Median: {0:.2f}{1}".format(ser_median, units), fontsize=14)

    if highlight:
        ax.axvline(highlight[0], alpha=0.5)
        ax.text(highlight[0], series_to_plot.max() / 2,
                highlight[0], ha='right',
                fontsize=14)
        ax.axvline(highlight[1], alpha=0.5)
        ax.text(highlight[1], series_to_plot.min() / 2,
                highlight[1], ha='left',
                fontsize=14)
        ax.fill_between(highlight, series_to_plot.min() * 1.1,
                        series_to_plot.max() * 1.1, color='orange', alpha=0.2)

    # set axis parameters
    ax.set_title(plot_title, fontdict=font)
    ax.set_xlabel(xlabel, fontdict=font)
    ax.set_ylabel(ylabel, fontdict=font)
    ax.tick_params(labelsize=tick_label_size)

    if show_plot:
        plt.show()
    return


def plot_scatter(ser1=None, ser2=None,
                 ser1_name="x", ser2_name="y",
                 plot_title="", tick_label_size=14,
                 fit_reg=False, alpha=0.5):
    """
    a function to plot a scatter plot of 2 variables
    found in 'col1' and 'col2' of the
    supplied DataFrame 'df'
    """
    # set font parameters
    font = dict(family='serif', color='darkred', weight='normal', size=16)

    # set figure size
    plt.figure(figsize=(8, 8))

    # plot the scatter plot
    sns.regplot(x=ser1, y=ser2,
                fit_reg=fit_reg,
                scatter_kws={'alpha': alpha})

    # set axis parameters
    plt.ylabel(ser2_name, fontdict=font)
    plt.xlabel(ser1_name, fontdict=font)
    plt.title(plot_title, fontdict=font)
    plt.tick_params(labelsize=tick_label_size)

    plt.show()
    return


def plot_set_mean_median(set_to_plot, title,
                         mean_lift=1.1, median_lift=1.1,
                         legend_loc='best'):
    """
    function to plot a set, its mean, and median

    Input arguments: set_to_plot -- list   -- set of numbers to plot
                     title       -- string -- title of the plot
                     mean_lift   -- float  -- lift of text above mean line on the plot (default=1.1)
                     median_lift -- float  -- lift of text above median line on the plot (default=1.1)
                     legend_loc  -- string -- location of the legend for ax.legend (default='best')

    Output:          None, plots set, its mean, and median
    """
    # calculate mean and median of the input set
    set_mean = stats.mean(set_to_plot)
    set_median = stats.median(set_to_plot)

    # create a new axis
    ax = plt.axes(facecolor='whitesmoke')

    # plot the set
    ax.plot(set_to_plot, linestyle=':', linewidth=0.5, color='blueviolet', marker='o', label='Set numbers')

    # plot mean and median
    ax.axhline(set_mean, linestyle='--', linewidth=1.2, color='deeppink', label='Mean')
    ax.text(0, set_mean * mean_lift, "Mean of the set: {0:.2f}".format(set_mean))
    ax.axhline(set_median, linestyle=':', color='coral', label='Median')
    ax.text(0, set_median * median_lift, "Median of the set: {0:.2f}".format(set_median))

    # set axis parameters
    ax.set_title(title)
    ax.legend(loc=legend_loc, fancybox=True)
    plt.ylabel('Value')
    plt.xlabel('Element of the set')
    ax.grid(False)

    plt.show()
    return


def plot_bars_with_minmax(series_to_plot, title, horizontal=False, color='gray',
                          xlabel="", ylabel="", tick_label_size=16,
                          with_minmax=True, with_mean=True,
                          min_border=0.95, max_border=0.99, minmax_width=0.5,
                          sup_line=None):
    """
    function to plot a gray bar chart from a pandas Series,
    plots mean
    highlights bars with minimum values

    Input arguments: series_to_plot -- pandas Series -- Series to plot as a bar chart
                     title          -- string        -- string containing title of the chart
                     with_minmax    -- boolean       -- option to highlight extreme values
                                                        from the Series (default=True)
                     min_border     -- float         -- height limit for black min bar on the plot
                                                        (default=0.95)
                     max_border     -- float         -- height limit for lightgray max bar on the plot
                                                        (default=0.99)
                     minmax_width   -- float         -- width for min and max bars on the plot
                                                        (default=0.5)


    Output:          None, plots and shows bar chart with mean and extremes (optional) highlighted
    """
    font = dict(family='serif', color='darkred', weight='normal', size=16)

    # create figure and axis
    fig, ax = plt.subplots(1, figsize=(8, 5))

    if horizontal:
        # plot a horizontal bar chart from input Series
        plt.barh(y=series_to_plot.index,
                 width=series_to_plot,
                 color=color)
        if with_mean:
            # plot mean of the series
            ax.axvline(series_to_plot.mean(), color='black', linestyle='--', linewidth=1)
            ax.text(series_to_plot.mean() * 1.05,
                    0,
                    "Mean: {0:.2f}".format(series_to_plot.mean()),
                    fontsize=16)
        if sup_line:
            ax.axhline(sup_line, color='black', linestyle='--', linewidth=1)
        # optional: highlight the bars with minimum values (if input parameter 'with_min' is True)
        if with_minmax:
            min_se = series_to_plot[series_to_plot == series_to_plot.min()]
            plt.barh(min_se.index,
                     min_se * min_border,
                     minmax_width,
                     color='black')

            max_se = series_to_plot[series_to_plot == series_to_plot.max()]
            plt.barh(max_se.index,
                     max_se * max_border,
                     minmax_width,
                     color='lightgray')
        # set axis parameters
        ax.set_xlabel(xlabel, fontdict=font)
        ax.set_ylabel(ylabel, fontdict=font)
        ax.tick_params('both', labelrotation=1, labelsize=tick_label_size)

    else:
        # plot a bar chart from input Series
        plt.bar(x=series_to_plot.index,
                height=series_to_plot,
                color=color)

        if with_mean:
            # plot mean of the series
            ax.axhline(series_to_plot.mean(), color='black', linestyle='--', linewidth=1)
            ax.text(0,
                    series_to_plot.mean() * 1.01,
                    "Mean: {0:.2f}".format(series_to_plot.mean()))

        # optional: highlight the bars with minimum values (if input parameter 'with_min' is True)
        if with_minmax:
            min_se = series_to_plot[series_to_plot == series_to_plot.min()]
            plt.bar(x=min_se.index,
                    height=min_se * min_border,
                    color='black',
                    width=minmax_width)
            max_se = series_to_plot[series_to_plot == series_to_plot.max()]
            plt.bar(x=max_se.index,
                    height=max_se * max_border,
                    color='lightgray',
                    width=minmax_width)
        if sup_line:
            ax.axvline(sup_line, color='black', linestyle='--', linewidth=1)
        # set axis parameters
        ax.set_xlabel(xlabel, fontdict=font)
        ax.set_xticks(series_to_plot.index)
        ax.set_ylabel(ylabel, fontdict=font)
        ax.tick_params('both', labelrotation=1, labelsize=tick_label_size)

    # set general axis parameters
    ax.set_title(title, fontdict=font)
    ax.grid(False)

    plt.show()
    return


def plot_xy(x=None, y=None, line_color='blue',
            line_label="", line_label_size=16,
            line_label_xlift=1, line_label_ylift=1.05,
            points_x=None, points_y=None,
            points_color='red', points_size=3,
            plot_title="",
            v_line=None, h_line=None,
            sup_line_style='--', sup_line_width=0.5, sup_line_color='gray',
            x_lim=None, y_lim=None,
            create_plot=True, show_plot=True, ax=None):
    """
    plot funcs
    """

    def plot_lines(line_type, sup_line):
        """
        plots and annotates horizontal and vertical lines
        """
        if line_type == 'h':
            ax.axhline(sup_line, color=sup_line_color,
                       linestyle=sup_line_style, linewidth=sup_line_width)
            ax.text(0, sup_line, sup_line)
        elif line_type == 'v':
            ax.axvline(sup_line, color=sup_line_color,
                       linestyle=sup_line_style, linewidth=sup_line_width)
            ax.text(sup_line, 0, sup_line)
        else:
            print("func 'plot_lines': Input parameter 'line_type' must be either 'h' or 'v'.")

    line_text_style = dict(horizontalalignment='right',
                           verticalalignment='center',
                           fontsize=line_label_size,
                           fontdict={'family': 'monospace'})

    if create_plot:
        f, ax = plt.subplots(1, figsize=(6, 6))
        f.suptitle(plot_title)
        # plot axes
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ax.set_ylabel('y', rotation=0)
        ax.axhline(0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('x')
        ax.grid(False)

    # plot the function (if supplied in arguments)
    if x is not None and y is not None:
        ax.plot(x, y, color=line_color)

        ax.text(x[int(len(x) / 2)] * line_label_xlift,
                y[int(len(y) / 2)] * line_label_ylift,
                line_label, color=line_color, **line_text_style)

    # plot the points (if supplied in arguments)
    if points_x is not None and points_y is not None:
        ax.scatter(points_x, points_y, s=points_size, color=points_color)

    # plot vertical lines (if supplied)
    if isinstance(v_line, Iterable):
        for line in v_line:
            plot_lines('v', line)
    elif type(v_line) is float or type(v_line) is int \
            or type(h_line) is np.float64:
        plot_lines('v', v_line)

    # plot horizontal lines
    if isinstance(h_line, Iterable):
        for line in h_line:
            plot_lines('h', line)
    elif type(h_line) is float or type(h_line) is int \
            or type(h_line) is np.float64:
        plot_lines('h', h_line)

    # zoom axes
    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])

    if show_plot:
        plt.show()


def plot_df(df_to_plot, x_key, title, x_lim=None, y_lim=None):
    """
    function to plot a DataFrame, uses 'x_key' for x-axis

    -----------------------------------------------------
    Input arguments: df_to_plot    -- pandas DataFrame -- DataFrame with results to be plotted
                     x_key -- string           -- index of the column to be used for x axis
                     x_lim -- list of 2 floats -- list of 2 limits to be used on the x axis of the plot (default=None)
                     y_lim -- list of 2 floats -- list of 2 limits to be used on the y axis of the plot (default=None)

    Returns:         None, plots a graph of a function
    """
    # create figure and axis
    f, ax = plt.subplots(1, figsize=(8, 8))

    # plot results
    df_to_plot.plot(x=x_key, ax=ax)

    # set axis parameters
    ax.set_title(title)
    ax.axhline(0, linestyle='--', linewidth=1, color='black')
    ax.axvline(0, linestyle='--', linewidth=1, color='black')
    ax.set_xticks(np.arange(df_to_plot[x_key].min(), df_to_plot[x_key].max() + 1))
    ax.grid(False)

    # set axis limits (if supplied)
    if x_lim is not None:
        ax.set_xlim(x_lim)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    plt.show()


# -------------------------- Math ---------------------------------------


def f(x):
    """ placeholder for a function f(x) """
    return x


def df(x, delta=0.00001):
    """ derivative of a function """
    return (f(x + delta) - f(x)) / delta


def ddf(x, delta=0.00001):
    """ second derivative of a function """
    return (df(x + delta) - df(x)) / delta


def solve(func, value, x=0.5, delta=0.00001,
          max_tries=1000, max_err=0.1,
          print_all=False, x_round=5):
    """
    equation solver
    find 'x' that maps through 'func' to 'value'

    moves 'x' by 'err' divided by 'slope'
    in the direction opposite to 'slope'
    until 'func(x)' - 'value' < 'max_err'

    """
    for tries in range(max_tries):
        err = func(x) - value
        if abs(err) < max_err:
            print("Solved in {0:,} steps!".format(tries))
            return x
        slope = (func(x + delta) - func(x)) / delta
        x = x - err / slope
        if print_all:
            print(x)
    print("After {0:,} iterations, no solutions found within:\nerr = {1}"
          .format(max_tries, max_err))
    print("Last value of x = {0:.{1}f}".format(x, x_round))
    return


def taylor_approx_sin_x(x, terms):
    """
    function that uses Taylor Series expansion of sin(x)
    to find approximate values for supplied range of x

    -----------------------------------------------------
    Input arguments: x         -- np.array -- array with x values
                     n         -- integer  -- number of terms in the Taylor Series expansion
                                              (x - x^3/3! + x^5/5! + ...)

    Returns: approximation     -- float    -- approximated values of the function sin(x)
    """
    # create an numpy array to store results
    approximation = np.array(0)

    # perform Taylor Series expansion: e^x = 1 + x+ x^2 / 2! + x^3 / 3! + ...
    # loop over all terms (subtract 1 for the first term)
    for n in np.arange(terms):
        # add subsequent terms (+1 to start from 1, not 0)
        approximation += (-1) ** n / np.math.factorial(2 * n + 1) * x ** (2 * n + 1)
    # return an array with approximated values
    return approximation


def taylor_sin_x_loop(min_x, max_x, min_terms, max_terms):
    """
    function that loops the Taylor Series approximation
    increasing number of terms used in approximation

    calls another function 'taylor_ser_approx_sin_x'
    and returns a DataFrame with results

    Input arguments: min_x            -- float            -- minimum value of x to be used for approximation
                     max_x            -- float            -- maximum value of x to be used for approximation
                     min_terms        -- integer          -- minimum number of terms to be used in the
                                                             approximation using Taylor Series expansion
                     max_terms        -- integer          -- maximum number of terms to be used in the
                                                             approximation using Taylor Series expansion

    Returns:         ts_df            -- pandas DataFrame -- DataFrame with results of approximation
    """
    # create an empty DataFrame to store results
    ts_df = pd.DataFrame()
    # create x-axis values
    ts_df['x'] = np.linspace(min_x, max_x, 1000)
    # calculate values of the actual function
    ts_df['f(x)'] = np.sin(ts_df['x'])

    # loop over the range of terms for Taylor Series expansion
    for terms in np.arange(min_terms, max_terms + 1):
        # call function that performs Taylor Series expansion, supplying x argument and the number of terms
        ts_df['ts_approx_' + str(terms)] = taylor_approx_sin_x(ts_df['x'], terms)

    # return DataFrame with results of Taylor Series expansion
    return ts_df


# ---------------------- Data Cleaning ----------------------------------


def unique_values(df_column: pd.Series, value_counts=True):
    """
    function to check unique values in a data frame column
    :param df_column: pd.Series -- column (pandas Series) to be checked for unique values
    :param value_counts: boolean -- (optional) show value counts, (default=True)
    :return: None
    """
    column_total_values: int = len(df_column)
    column_unique_values: int = len(df_column.value_counts())
    print("Out of the total {0:,} records in column '{1}', {2:,}({3:.2f}%) have unique values."
          .format(column_total_values,
                  df_column.name,
                  column_unique_values,
                  column_unique_values / column_total_values * 100))
    if value_counts:
        print("\nValue counts for column '{0}':".format(df_column.name))
        print(df_column.value_counts())


def duplicate_check(df, subsets_to_check: dict = None, cols_to_drop=None):
    """
     function to perform the duplicate check

     takes as input a DataFrame and one (or both) of the optional parameters:

     'subsets_to_check' is a dict that will be used for duplicate check criteria
     its keys represent the name of the subset of columns and its values represent
     the lists of columns to be used as criteria

     'subset_to_drop' is a list of columns to be excluded one by one from criteria
     to perform duplicate checks without these columns
     if 'subset_to_drop' is empty, this step is skipped

    :param df: pd.DataFrame -- DataFrame on which to perform the duplicate check

    :param subsets_to_check: dict -- dict, where keys are column subset names and
                                     values are subsets of columns to use

    :param cols_to_drop: Iterable -- an iterable containing names of columns to be
                                     excluded one by one from detection criteria

    :return dup_result_df: pd.DataFrame -- DataFrame with results of the duplicate check
    """

    def perform_duplicate_check(result_key, col_subset):
        """
        function to perform the actual duplicate check
        and record its results

        :param result_key: string -- key to be used to record results of the check
        :param col_subset: list -- subset that will be used to perform the check

        :return: None, updates nonlocal dict 'dup_results'
        """
        # variables from the outer scope
        nonlocal df, dup_results
        # create a new entry in results dictionary 'dup_results'
        dup_results[result_key] = dict()
        # determine the number of duplicates
        dup_results[result_key]['num_duplicates'] = df.duplicated(subset=col_subset).sum()
        dup_results[result_key]['num_total'] = len(df)
        dup_results[result_key]['percentage'] = dup_results[result_key]['num_duplicates'] \
                                                / dup_results[result_key]['num_total'] * 100
        print("Subset '{0}': {1:,} ({2:.2f}% of total {3:,}) records are detected as duplicated."
              .format(result_key,
                      dup_results[result_key]['num_duplicates'],
                      dup_results[result_key]['percentage'],
                      dup_results[result_key]['num_total']))

    # create a dict to store results of the duplicate check
    dup_results = dict()

    # duplicate check using all columns as criteria
    key = 'all_columns'
    subset = df.columns
    perform_duplicate_check(key, subset)

    # duplicate check using 'cols_to_drop' -- each test takes all columns minus one
    if isinstance(cols_to_drop, Iterable):
        for col in cols_to_drop:
            key = col
            subset = df.columns
            subset = subset.drop(col)
            perform_duplicate_check(col, subset)

    if isinstance(subsets_to_check, dict):
        for k, v in subsets_to_check.items():
            perform_duplicate_check(k, v)

    dup_results_df = pd.DataFrame(dup_results)

    # plot results of the check
    # create figure and axis
    f, ax = plt.subplots(1, figsize=(6, 6))

    # plot results
    dup_results_df.T['num_duplicates'] \
        .sort_values() \
        .plot(kind='barh', color='gray')

    # set axis parameters
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    ax.set_title("Results of the duplicate check", fontdict=font)
    ax.set_ylabel("Subset used as detection criteria", fontdict=font)
    ax.set_xlabel("Number of duplicate records", fontdict=font)

    ax.tick_params(axis='both', labelsize=14)

    return dup_results_df


# ---------------------- Text Processing ----------------------------------


def tokenize_text(text):
    """
    tokenizes and filters provided text into words

    since this function is used inside of a pandas.apply
    method, it can only accept one parameter 'text' as
    input, and therefore relies on the rest of variables
    to be declared outside of the function and accessed
    through the global scope

    these variables include: NLTK tokenizer as 'tokenizer',
    NLTK stemmer as 'stemmer', list of stop words as 'stop_words',
    and min length used to filter tokens as 'min_length'

    first, tokens are created from input text
    using the global tokenizer 'tokenizer',

    then, words are converted to lower case,
    punctuation and stop words are removed,

    after that, words are stemmed using the
    global stemmer 'stemmer'

    finally, all words with length < 3 are filtered out

    param: text       -- string -- text to be tokenized

    global variable: tokenizer  -- NLTK tokenizer
                                   tokenizer used to generate tokens
                                   (needs to be initialized as 'tokenizer'
                                   before this function can be called)
    global variable: stemmer    -- NLTK stemmer
                                   stemmer used to stem tokens
                                   (needs to be initialized as 'stemmer'
                                   before this function can be called)
    global variable: stop_words -- list
                                   list of stop words to use for filtering
                                   out all tokens that are in this list
    global variable: min_length -- int
                                   min length used to filter out all
                                   tokens shorter than 'min_length'

    returns: filtered_tokens -- list of tokens produced from text
    """
    # tokenizer and stemmer used by the function, need to be initialized prior to call
    global tokenizer, stemmer, stop_words, min_length
    # creating a map object with 'text' tokenized into lower-cased 'words'
    words = map(lambda word: word.lower(), tokenizer.tokenize(text))

    # remove stop words -- common words, such as 'the', 'a', 'in', etc. using the provided list
    words = [word for word in words if word not in stop_words]

    # stem the words
    tokens = (list(map(lambda token: stemmer.stem(token), words)))

    # remove all the words with length less than 3
    filtered_tokens = list(filter(lambda token: len(token) >= min_length, tokens))

    return filtered_tokens


def labelize_tweets_bg(token_series, tweets, label):
    """
    a function to transform a corpus
    using the bigram model that will
    detect frequently used phrases of
    two words, and stick them together
    """
    phrases = Phrases(token_series)
    bigram = Phraser(phrases)
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(LabeledSentence(bigram[t],
                                      [prefix + '_%s' % i]))
    return result


def string_concat(ser, string_name="", display_sym=500,
                  input_type='strings'):
    """
    a function to concatenate a single string
    out of a series of text documents
    """
    con_string = []

    if input_type == 'strings':
        for text in ser:
            con_string.append(text)

    elif input_type == 'lists':
        for str_list in ser:
            con_string.append(" ".join(str_list))
    else:
        print("'input_type' must be 'strings' or 'lists'.")

    con_string = pd.Series(con_string).str.cat(sep=' ')

    print("First {0} symbols in the {1} string:\n"
          .format(display_sym, string_name))
    print(con_string[:display_sym])

    return con_string


def tfm_2class(df, label_col, label_vals, text_col,
               class_names=('neg_tf', 'pos_tf'),
               sw='english',
               min_df=0.01, max_df=0.9,
               return_type='tfm'):
    """
    a function to create a Term Frequency Matrix
    from the corpus of documents found in
    column 'text_col' of DataFrame 'df'

    this function is designed to work with 2 classes
    of target variable found in column 'label_col' of 'df',
    values of classes (e.g., -1, 1)
    need to be supplied as a list in parameter 'label_vals'

    class names for the return DataFrame
    can be specified via parameters
    'class1_name' and 'class2_name'

    'text_col' is the name of column in 'df'
    containing documents to be summarized

    param: df -- pd.DataFrame  -- DataFrame that contains
                                the corpus to be summarized
           label_col -- string -- name of the column
                                in 'df' containing label
                                (target) information
           label_vals -- list  -- list of values that
                                 'label_col' can take
                                 (e.g., [1, -1])
                                 only 2 values supported

    returns: tfm_df -- pd.DataFrame -- term frequency
                                       matrix of the corpus
    """
    # initialize CountVectorizer from Scikit-learn
    vectorizer = CountVectorizer(strip_accents='unicode',
                                 stop_words=sw,
                                 max_df=max_df,
                                 min_df=min_df)

    # fit vectorizer to corpus in 'text_col' of 'df'
    vectors_f = vectorizer.fit(df[text_col])

    # create a subset of 'df' with all records of class 1
    class1_subset = df \
        .loc[df[label_col] == label_vals[0], text_col]
    # vectorize subset into a sparse matrix
    class1_doc_matrix = vectors_f \
        .transform(class1_subset)

    # create a subset of 'df' with all records of class 2
    class2_subset = df \
        .loc[df[label_col] == label_vals[1], text_col]
    # vectorize subset into a sparse matrix
    class2_doc_matrix = vectors_f \
        .transform(class2_subset)

    # sum occurrence of each token
    class1_tf = np.sum(class1_doc_matrix, axis=0)
    class2_tf = np.sum(class2_doc_matrix, axis=0)

    # remove single-dimensional entries from the shape of the arrays
    class1 = np.squeeze(np.asarray(class1_tf))
    class2 = np.squeeze(np.asarray(class2_tf))

    # create a DataFrame with token frequencies by class
    tfm_df = pd.DataFrame([class1, class2],
                          columns=vectors_f.get_feature_names()) \
        .transpose()

    # change column names
    tfm_df.columns = class_names

    # create a new column with total token frequency
    tfm_df['total'] = tfm_df[class_names[0]] \
                      + tfm_df[class_names[1]]

    # create a new column with difference between classes
    tfm_df['abs_diff'] = \
        abs(tfm_df[class_names[0]]
            - tfm_df[class_names[1]])

    if return_type == 'tfm':
        # return Term Frequency Matrix for the corpus
        return tfm_df

    elif return_type == 'dtr':
        # return sum of abs of all diff by total sum
        return tfm_df['abs_diff'].sum() / tfm_df['total'].sum()
    else:
        print("'return_type' must be either 'tfm' " +
              "for Term Frequency Matrix")
        print("or 'dtr' for AbsDiff / Total ratio.")


def plot_wordcloud(string, colormap='viridis'):
    """
    a function to plot a Word Cloud from a string of tokens
    """
    wordcloud = WordCloud(width=1600,
                          height=800,
                          max_font_size=200,
                          colormap=colormap).generate(string)

    f, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# -------------------- Train / Test split --------------------------------


def train_dev_test_split(x, y, label_values,
                         test_size=.02,
                         random_state=2000):
    """
    a function to split data (x and y)
    into three chunks: train, development, test.

    The data is split using the `train_test_split`
    function from `scikit-learn` two times:

    first, split train and dev+test;

    then, split dev and test from dev+test.
    """
    # split data into train and dev+test subsets
    x_train, x_validation_and_test, \
    y_train, y_validation_and_test = \
        train_test_split(x, y,
                         test_size=test_size,
                         random_state=random_state)

    # split dev+test subset into dev and test subsets
    x_validation, x_test, y_validation, y_test = \
        train_test_split(x_validation_and_test,
                         y_validation_and_test,
                         test_size=.5,
                         random_state=random_state)

    print("Train set has total {0} entries"
          .format(len(x_train)))
    print("with {0:.2f}% negative, {1:.2f}% positive.\n"
          .format((len(x_train[y_train == label_values[0]])
                   / (len(x_train) * 1.)) * 100,
                  (len(x_train[y_train == label_values[1]])
                   / (len(x_train) * 1.)) * 100))

    print("Validation set has total {0} entries"
          .format(len(x_validation)))
    print("with {0:.2f}% negative, {1:.2f}% positive.\n"
        .format(
        (len(x_validation[y_validation == label_values[0]])
         / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == label_values[1]])
         / (len(x_validation)*1.))*100))

    print("Test set has total {0} entries"
          .format(len(x_test)))
    print("with {0:.2f}% negative, {1:.2f}% positive."
        .format(
        (len(x_test[y_test == label_values[0]])
         / (len(x_test)*1.))*100,
        (len(x_test[y_test == label_values[1]])
         / (len(x_test)*1.))*100))

    # return 6 subsets -- train, dev, and test x and y
    return x_train, x_validation, x_test, \
           y_train, y_validation, y_test


def train_test_split_temp(input_data, train_subset_ratio):
    """
    a function to perform a temporal train test split
    :param input_data:  -- data to be split
    :param train_subset_ratio:  -- ratio to used to generate training subset
    :return: train -- training subset
             test  -- testing subset

    """
    # set train subset ratio
    train_size = int(len(input_data) * train_subset_ratio)

    # split the data set into train and test
    train, test = input_data[0:train_size], input_data[train_size:len(input_data)]
    print('Observations: %d' % (len(input_data)))
    print("\nTrain_test split ratio: {0:.2f}%".format(train_subset_ratio * 100))
    print('\nTraining Observations: %d' % (len(train)))
    print('Testing Observations: %d' % (len(test)))

    return train, test


def plot_split(train, test, plot_title="", ylabel="y",
               train_caption_lift=2.5, test_caption_lift=2.5):
    """
    a function to plot the train-test split
    :param train:                     -- train subset
    :param test:                      -- test subset
    :param plot_title:                -- title of the plot
    :param ylabel:                    -- label for y axis
    :param test_caption_lift:         -- lift of test line label on the plot
    :param train_caption_lift:        -- lift of train line label on the plot
    :return:
    """
    # plot train data
    plot_time_series(train, plot_title=plot_title,
                     ylabel=ylabel, y_log=True,
                     alpha=0.4, mean=False, median=False, minmax=False, show_plot=False)

    # get axis generated by the previous call to the plotting function
    ax = plt.gca()

    # plot test data on the same axis
    plot_time_series(test, color='green',
                     plot_title=plot_title, ylabel=ylabel,
                     alpha=0.4, mean=False, median=False, minmax=False,
                     create_plot=False, show_plot=False, ax=ax)

    # plot split line
    ax.axvline(test.index[0], linestyle='--', color='black')

    # add captions
    ax.text(train.index[len(train) // 3], train.mean() * train_caption_lift,
            'Training data', color='blue', fontsize=16)
    ax.text(test.index[len(test) // 3], test.mean() * test_caption_lift,
            'Test data', color='darkgreen', fontsize=16)

    plt.show()
    return


# ----------------- Model training and validation -----------------------


def model_performance_report(labels,
                             predictions,
                             label_values):
    """
    a function to assess model performance
    in predicting labels and print a report
    """
    con_mat = np.array(confusion_matrix(labels,
                                        predictions,
                                        label_values))

    confusion = pd.DataFrame(con_mat,
                             index=['positive', 'negative'],
                             columns=['predicted_positive',
                                      'predicted_negative'])

    print("")
    print("Accuracy Score: {0:.2f}%"
          .format(accuracy_score(labels, predictions)*100))
    print("-"*80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-"*80)
    print("Classification Report\n")
    print(classification_report(labels, predictions))
    return


def limit_accuracy_checker(classifier, limits,
                           x_train, x_validation,
                           y_train, y_validation,
                           vectorizer_type='freq',
                           sw='english',
                           ngram_range=(1, 1),
                           label_values=(0, 1)):
    result = []
    print(classifier)
    print("\n")

    for limit in limits:

        print("----- '{0}' vectorization"
              .format(vectorizer_type.upper()))
        print("term document frq filter: {1} min {2} max"
              .format(vectorizer_type.upper(),
                      limit[1],
                      limit[0]))
        print("stop words:", sw)

        if vectorizer_type == 'freq':
            # vectorize using frequency encoding
            vectorizer = \
                CountVectorizer(binary=False,
                                strip_accents='unicode',
                                stop_words=sw,
                                max_df=limit[0],
                                min_df=limit[1],
                                ngram_range=ngram_range)

        elif vectorizer_type == 'onehot':
            # vectorize using one-hot encoding
            vectorizer = \
                CountVectorizer(binary=True,
                                strip_accents='unicode',
                                stop_words=sw,
                                max_df=limit[0],
                                min_df=limit[1],
                                ngram_range=ngram_range)

        elif vectorizer_type == 'tfidf_freq':
            # vectorize using TF-IDF frequency encoding
            vectorizer = \
                TfidfVectorizer(binary=False,
                                strip_accents='unicode',
                                stop_words=sw,
                                max_df=limit[0],
                                min_df=limit[1],
                                ngram_range=ngram_range)

        elif vectorizer_type == 'tfidf_onehot':
            # vectorize using TF-IDF one-hot vectors
            vectorizer = \
                TfidfVectorizer(binary=True,
                                strip_accents='unicode',
                                stop_words=sw,
                                max_df=limit[0],
                                min_df=limit[1],
                                ngram_range=ngram_range)
        else:
            raise ValueError("'vectorizer_type' must be 'freq', 'onehot', 'tfidf-freq', or 'tfidf-onehot'")

        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        # assess model performance
        limit_accuracy, tt_time = \
            accuracy_summary(checker_pipeline,
                             x_train,
                             y_train,
                             x_validation,
                             y_validation,
                             label_values)

        # add model performance to results
        result.append((limit, limit_accuracy, tt_time))
    return result


def accuracy_summary(pipeline,
                     x_train, y_train,
                     x_test, y_test,
                     label_values=(0, 1)):

    # null accuracy using the Zero Rule
    if len(x_test[y_test == label_values[0]]) \
            / (len(x_test)*1.) > 0.5:
        # predicting majority class
        null_accuracy = \
            len(x_test[y_test == label_values[0]]) \
            / (len(x_test)*1.)
    else:
        null_accuracy = \
            1. - (len(x_test[y_test == label_values[0]])
                  / (len(x_test)*1.))

    # set starting time
    t0 = time()

    # fit the model
    sentiment_fit = pipeline.fit(x_train, y_train)

    # test the model
    y_pred = sentiment_fit.predict(x_test)

    # record train-test time
    train_test_time = time() - t0

    # compute accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # print model performance results
    print("null accuracy: {0:.2f}%"
          .format(null_accuracy*100))
    print("accuracy score: {0:.2f}%"
          .format(accuracy*100))

    # compare model performance to null accuracy
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate \
than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with \
the null accuracy")
    else:
        print("model is {0:.2f}% less accurate \
than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s"
          .format(train_test_time))
    print("-"*80)
    return accuracy, train_test_time


def plot_model_results(model_results_df,
                       x, y, hue=None,
                       xlabel="", ylabel="",
                       title="",
                       null_accuracy=None,
                       plot_max=True,
                       text_lift=1.03):
    """
    a function to plot results of model performance.
    takes in a melted DataFrame with model results
    plots bar charts grouped by 'x' colored by 'hue'

    for the plotting function to work, each row
    in the DataFrame must correspond to 1 bar only
    (DataFrame must be melted)
    """
    # font to be used in axes labels
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}
    # font to be used for null accuracy
    font_null_acc = {'family': 'serif',
                     'color':  'black',
                     'weight': 'normal',
                     'size': 16}
    # font to be used for model accuracy
    font_acc = {'family': 'serif',
                'color':  'darkgreen',
                'weight': 'normal',
                'size': 16}

    # plot grouped bar chart
    sns.catplot(x=x, y=y, hue=hue, data=model_results_df, kind='bar')
    # get axis created by seaborn
    ax = plt.gca()

    if plot_max:
        # plot max value from model performance
        ax.axhline(model_results_df[y].max(),
                   color='darkgreen',
                   linestyle='--',
                   linewidth=2)
        ax.text(0, model_results_df[y].max() * text_lift,
                "Best accuracy: {0:.2f}"
                .format(model_results_df[y].max()),
                fontdict=font_acc)

    if null_accuracy:
        # plot null accuracy
        ax.axhline(null_accuracy,
                   linestyle='--',
                   color='black',
                   linewidth=2)
        ax.text(0, null_accuracy * text_lift,
                "Null accuracy: {0:.2f}"
                .format(null_accuracy),
                fontdict=font_null_acc)

    # set axis parameters
    ax.set_ylabel(ylabel, fontdict=font)
    ax.set_xlabel(xlabel, fontdict=font)
    ax.set_title(title)
    plt.show()


# -------------------- Specific functions -------------------------------


def plot_q_parts(question_list, response_df,
                 question, parts,
                 comment='', return_subset=False, tick_label_size=12):
    """
    a function to plot results of a question from Kaggle survey
    that are stored in a range of columns
    """
    question_start_col = question + '_Part_1'
    question_last_col = question + '_Part_' + str(parts)

    question_subset = response_df.loc[:, question_start_col:question_last_col]
    answer_categories = []
    for column in question_subset.columns:
        answer_categories.append(question_list.loc[column].split(' - ')[-1])

    question_subset.columns = answer_categories

    series_to_plot = question_subset.count().sort_values(ascending=False)

    title = question + ': ' + question_list[question + '_Part_1'].split(' - ')[0] + '\n' + comment

    plot_bars_with_minmax(series_to_plot, title, horizontal=True,
                          with_mean=False,
                          tick_label_size=tick_label_size)

    if return_subset:
        return series_to_plot
    else:
        return

# UNFINISHED FUNCTIONS
# #def plot_dist(ser=None, df=None,
# #              hist=True, bins=10, kde=True, rug=True,
#               vertical=False,
#               create_plot=True, show_plot=True, ax=None):
#     """
#     a function to plot distribution plot of the provided Series,
#     or distribution plots of all Series in the provided DataFrame
#     :param ser:            pandas.Series -- Series to be plotted
#     :param df:
#     :param hist:
#     :param bins:
#     :param kde:
#     :param rug:
#     :param vertical:
#     :param create_plot:    boolean       -- whether to create figure and axis
#                                             (set to False for subsequent plots on same axis)
#     :param show_plot:      boolean       -- whether to show the plot
#                                             (set to False for subsequent plots on same axis)
#     :param ax:          matplotlib axis  -- if provided, plot on this axis
#                                             (if subsequent plot, provide ax)
#     :return:
#     """
#     if ser:
#
#         # plot the distribution plot for the provided Series
#         sns.distplot(ser, hist=hist, kde=kde, bins=bins, rug=rug, vertical=vertical)
#
#     if df:
#
#         # loop over all columns in the DataFrame
#         for column in df.columns:
#             # plot the distribution plot for each column
#             sns.distplot(column, hist=hist, kde=kde, bins=bins, rug=rug, vertical=vertical)
#
#     plt.show()