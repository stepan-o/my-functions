import matplotlib.pyplot as plt
import seaborn as sns


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


def train_test_split(input_data, train_subset_ratio):
    """

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