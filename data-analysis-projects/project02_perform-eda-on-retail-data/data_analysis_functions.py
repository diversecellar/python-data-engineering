# Import common df functions
import matplotlib
import seaborn as sns
import statsmodels

### CODE ###
# load all fundamental df functions
# list of all df functions
### YOUR CODE HERE ### 
def df_print_row_and_columns(df_name):
    df_rows, df_columns = df_name.shape
    print("rows = {}".format(df_rows))
    print("columns = {}".format(df_columns))

def df_check_na_values(df_name, *args):
    if not args:
        df_na = df_name.isna()
        mask = df_na == True
        masked = df_na[mask]
    else:
        df_na = df_name.isna()
        try:
            column_names = [arg for arg in args[0] if (isinstance(arg, str) and isinstance(args[0], list))]
        except Exception as e:
            print("need to be list of str type for args")
        for column in column_names:
            mask = df_na[column] == True
            masked = df_na[mask]
    print(masked)
    return df_name.isna()

def df_drop_na(df_name, ax: int):
    if ax in [0, 1]:
        #df_na_out = df_name[~df_null]
        df_na_out = df_name.dropna(axis=ax)
        return df_na_out

def df_datetime_converter(df_name, col_datetime_lookup='date'):
    for column in df_name.columns.tolist():
        if str(col_datetime_lookup) in str(column):
            print("yes")
            df_name[column] = pd.to_datetime(df_name[column])
    return df_name

def df_boxplotter(df_name, col_xplot, col_yplot, type_plot:int, *args): #type_plot -> 0 for dist, 1 for money
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6), dpi=100)
    # Initiate the sns boxplotter
    sns.boxplot(x=df_name[col_xplot],
           y=df_name[col_yplot], ax=ax)
    # Set the plot title
    matplotlib.pyplot.title('{} box plot to visualise outliers'.format(col_yplot))
    # Here we set the y-axis (response) labels 
    if type_plot == 0:
        matplotlib.pyplot.ylabel('{} in miles'.format(col_yplot))
    if type_plot == 1:
        matplotlib.pyplot.ylabel('{} in $'.format(col_yplot))
    if type_plot == 2:
        matplotlib.pyplot.ylabel('{}'.format(col_yplot))
    # Set xtick rotation
    if args:
        matplotlib.pyplot.xticks(rotation=0, horizontalalignment=arg[0])	
    # Set y axis grid only
    ax.yaxis.grid(True)
    matplotlib.pyplot.savefig("Boxplot_x-{}_y-{}.png".format(col_xplot, col_yplot))
    matplotlib.pyplot.show()

def df_explore_unique_categories(df_name, col):
    # to print a df with unique categories for each cat var
    df_col_unqiue = df_name.drop_duplicates(subset=col, keep='first')
    return df_col_unqiue[col]

def df_histplotter(df_name, col_plot, type_plot:int, bins=10, *args): #type_plot -> 0 for dist, 1 for money
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6), dpi=85)
    x, y = sns.histplot(data=df_name, x=col_plot)
    matplotlib.pyplot.title('{} histogram plot'.format(col_plot))
    if type_plot == 0:
        matplotlib.pyplot.xlabel('{} in miles'.format(col_plot))
    if type_plot == 1:
        matplotlib.pyplot.xlabel('{} in $'.format(col_plot))
    if type_plot == 2:
        matplotlib.pyplot.xlabel('{}'.format(col_plot))
    #if df_name[col_plot].max() > 0:
    #    matplotlib.pyplot.xlim(0.0, 1.25*df_name[col_plot].max())
    #else:
    #    matplotlib.pyplot.xlim(0.0-(1.25*df_name[col_plot].max()/bins), 1.25*df_name[col_plot].max())
    if args:
        if isinstance(args[0], dict):
            args_dict = args[0]
            mean = ax.vlines(x=args_dict['mean'], ymin=y.min(), ymax=y.max(), colors='mean')
            mode = ax.vlines(x=args_dict['mode'], ymin=y.min(), ymax=y.max(), colors='mode')
            median = ax.vlines(x=args_dict['median'], ymin=y.min(), ymax=y.max(), colors='median')
    matplotlib.pyplot.savefig("Histogram_x-{}.png".format(col_plot))
    matplotlib.pyplot.show()

def df_grouped_histplotter(df_name, col_groupby: str, col_plot: str , type_plot: int, bins=20):
    df_vendor_grouped = df_name.groupby(['VendorID'])
    matplotlib.pyplot.figure(figsize=(8, 6), dpi=85)
    matplotlib.pyplot.title('{} by {} grouped hist plot'.format(col_plot, col_groupby))
    for key, item in df_vendor_grouped:
        group = df_vendor_grouped.get_group(key)
        matplotlib.pyplot.hist(group[col_plot], alpha=0.5, label=str(key), bins=bins)
    if type_plot == 0:
        matplotlib.pyplot.xlabel('{} in miles'.format(col_plot))
    if type_plot == 1:
        matplotlib.pyplot.xlabel('{} in $'.format(col_plot))
    matplotlib.pyplot.plot()
    matplotlib.pyplot.legend(title=col_groupby)
    matplotlib.pyplot.savefig("Grouped-Histogram_x-{}.png".format(col_plot))
    matplotlib.pyplot.show()

def df_mask_with_list(df, df_col, list_comp: list, mask_type: int): # mask_type is 0 or 1 (0 for not in, 1 for in)
    mask_if_industry_list = df[df_col].isin(list_comp)
    if bool(mask_type):
        companies_not_in_industry_list = df[mask_if_industry_list]
    else:
        companies_not_in_industry_list = df[~mask_if_industry_list]
    return print(list(set(companies_not_in_industry_list[df_col].tolist())))

def df_groupby_mask_operate(df, col_name_masker: str, col_name_operate: str, filter_bool: bool, filter_str: str, *args):
    df_groupby = df
    operations_list = []
    for arg in args:
        operations_list.append(arg)
    if filter_bool:
        if filter_str in payment_names:
            keys = list(filter(lambda key: payments_match_dict[key] == filter_str, payments_match_dict))
            mask = df_groupby[col_name_masker] == int(keys[0])
        else:
            keys = filter_str
            mask = df_groupby[col_name_masker] == keys
        df_filtered = df_groupby[mask]
        df_aggregate = df_filtered[[col_name_operate]].agg(operations_list)
        return df_aggregate[col_name_operate][0]
    else:
        df_filtered = df_groupby.groupby([col_name_masker]).agg({col_name_operate:operations_list})
        return df_filtered

def df_grouped_barplotter(df_name, col_groupby: str, col_plot: str, type_plot: int):
    df_grouped = df_groupby_mask_operate(df_name, col_groupby, col_plot, 0, '1', 'mean')
    x_plot = [row for row, index in df_grouped.iterrows()]
    y_plot = [index[0] for row, index in df_grouped.iterrows()]
    matplotlib.pyplot.figure(figsize=(8, 6), dpi=85)
    matplotlib.pyplot.title('{} by {} grouped bar plot'.format(col_plot, col_groupby))
    sns.barplot(x=x_plot, y=y_plot, ci=False)
    if type_plot == 0:
        matplotlib.pyplot.ylabel('{} in miles'.format(col_plot))
    if type_plot == 1:
        matplotlib.pyplot.ylabel('{} in $'.format(col_plot))
    matplotlib.pyplot.legend(title=col_groupby)
    if isinstance(x_plot[0], int) or isinstance(x_plot[0], float):
        matplotlib.pyplot.xticks(rotation=0)
    if isinstance(x_plot[0], str):
        matplotlib.pyplot.xticks(rotation=90)
    matplotlib.pyplot.ylim(0.0, 1.25*max(y_plot))
    matplotlib.pyplot.show()

def df_drop_dupes(df, col_dupes: int, *args): # col_dupes asks only for column duplicates and then *args is col_name
    if bool(col_dupes):
        mask_dupes = df.duplicated(subset=[args[0]],keep=args[1]) == True
    else:
        mask_dupes = df.duplicated(keep=args[1]) == True
    df_duplicated = df[~mask_dupes]
    return df_duplicated

def df_scatterplotter(df_grouped, col_xplot, col_yplot):
    matplotlib.pyplot.figure(figsize=(8, 6), dpi=85)
    sns.scatterplot(x=col_xplot, y=col_yplot, data=df_grouped)
    matplotlib.pyplot.title('{} by {} scatter plot'.format(col_xplot, col_yplot))
    matplotlib.pyplot.savefig("Scatter-Plot_x-{}_y-{}.png".format(col_xplot, col_yplot))
    matplotlib.pyplot.show()
    
def df_corr_check(df_name, col_y, col_x):
    # numerical linearity check using correlation coefficient
    correl = df_name[col_y].corr(df_name[col_x])
    print("{} is correlated to {} with a correl coef. r = {}".format(col_y, col_x, correl))

def df_gaussian_checks(df_name, col_name, *args):
    rules_dict = {1: 0.68,2: 0.95, 3: 0.997}
    conditions_dict = {}
    df_mean, df_stddev = (df_name[col_name].mean(), df_name[col_name].std()) 
    for key in rules_dict.keys():
        conditions_dict[key] = [df_mean]
        conditions_dict[key].append(df_mean + key*df_stddev)
        conditions_dict[key].append(df_mean - key*df_stddev)
    #print(conditions_dict) # [mean, *std_devs]
    #check_dict = {}
    if len(args) > 0:
        stop_key = args[0]
    else:
        stop_key = 3
    for key in conditions_dict.keys():
        while key <= stop_key:
            conditions_mask = ((df_name[col_name] <= conditions_dict[key][1]) 
                               & (df_name[col_name] >= conditions_dict[key][2]))
            df_conditioned = df_name[conditions_mask].agg(['count'])[col_name]
            df_unconditioned = df_name.agg(['count'])[col_name]
            rule = rules_dict[key]
            check = df_conditioned[0]/df_unconditioned[0]
            rule_in_pct = format(rule, ".1%")
            if check >= rule:
                print("check if {} of data-values within {} std_dev = fail"
                      .format(rule_in_pct, key))
            else:
                print("check if {} of data-values within {} std_dev = pass"
                      .format(rule_in_pct, key))
            break

def df_calc_conf_interval(moe_vals:dict, mean_val):
    conf_vals = {}
    for cl in moe_vals.keys():
        conf_vals[cl] = [np.round(mean_val - moe_vals[cl],2),\
                         np.round(mean_val + moe_vals[cl],2)]
        conf_level_pct = format(cl, ".0%")
        print("{} CI: {}".format(conf_level_pct, conf_vals[cl]))
    return conf_vals

# Lastly, use the preceding result to calculate your margin of error.
def df_calc_moe(stderr_val, z_score_cl):
    moe_vals = {}
    for cl in z_score_cl.keys():
        moe_vals[cl] = z_score_cl[cl] * stderr_val 
    return moe_vals

# Next, calculate your standard error.
def df_calc_stderr(df_name, col_z, stddev_val):
    col_z_len = int(df_name[col_z].shape[0])
    stderr_val = stddev_val/np.sqrt(col_z_len)
    return stderr_val

# Begin by identifying the z associated with your chosen confidence level.
def df_calc_zscore(df_name, col_z, confidence_levels, mean_val, stddev_val):
    col_z_len = int(df_name[col_z].shape[0])
    stderr_val = stddev_val/np.sqrt(col_z_len)
    df_name['z_score'] = (df_name[col_z] - mean_val)/stddev_val
    z_score_list = {}
    if isinstance(confidence_levels, list):
        for cl in confidence_levels:
            z_score_cl = df_name['z_score'].quantile(q=cl)
            z_score_list[cl] = z_score_cl
    elif isinstance(confidence_levels, float):
        cl = confidence_levels
        z_score_cl = df_name['z_score'].quantile(q=cl)
        z_score_list[cl] = z_score_cl
    return z_score_list

def df_head(df_name, head_num: int):
    print(df_name.head(head_num))
    
def df_pairplot(data):
    matplotlib.pyplot.figure(figsize=(6, 6), dpi=85)
    # Set the scatterplot matrix
    sns.pairplot(data)
    matplotlib.pyplot.show()
    
def lr_check_homoscedasticity(fitted, resid, *args):
    matplotlib.pyplot.figure(dpi=85)
    # Set the plot .
    fig_hsd = sns.scatterplot(fitted, resid)
    # Set the horizontal 0-line.
    fig_hsd.axhline(0.0)
    # Set the x-axis label.
    if len(args) > 0:
        fig_hsd.set_xlabel("Fitted {} Values".format(args[0]))
    else:
        fig_hsd.set_xlabel("Fitted Reponse Values")
    # Set the y-axis label.
    fig_hsd.set_ylabel("Residuals")
    # Set the plot title.
    fig_hsd.set_title("Checking Homescedasticity")

    matplotlib.pyplot.show()

def lr_check_normality(resid): # resid is the residuals as per the model
    # Create a 1x2 plot figures.
    matplotlib.pyplot.figure(figsize=(3,6), dpi=85)
    # Create a histogram with the residuals.
    fig_res = sns.histplot(resid)
    # Set the x label of the residual plot.
    fig_res.set_xlabel("Residual Value")
    # Set the title of the residual plot.
    fig_res.set_title("Historgram of Residuals/Errors to Check Normality")
    title = "Residuals"
    matplotlib.pyplot.savefig("LR-Histogram_x-{}.png".format(title))
    matplotlib.pyplot.show()

def lr_qqplots_normality(resid): # resid is the residuals as per the model
    # Create a Q-Q plot of the residuals.
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 5), dpi=85)
    fig_qq = smapi.qqplot(resid, line='s', ax=ax)
    # Set the title of the Q-Q plot.
    fig_qq.suptitle('Checking Residual/Error Normality using QQPlot')
    title = "Residuals"
    matplotlib.pyplot.savefig("LR-qqPlot_x-{}.png".format(title))
    matplotlib.pyplot.show()

def remove_whitespace(str_target: str):
    str_output = str_target.replace(" ","_")
    return str_output

def remove_unicode(str_target: str):
    regex_strings = ["[^\x00-\x7F]+", "[^\u0000-\u007F]"]
    for regex in regex_strings:
        if re.match(str_target, regex):
            str_out = unidecode(str_target)
        else:
            str_out = str_target
    return str_out

def lr_post_hoc_test(df_name, col_response, col_predictor, alpha:float):
    # col_response must be the continuous regressed/predicted variable
    # col_predictor must be categorical (groups) variable
    # alpha is significance level
    tukey_oneway = statsmodels.stats.multicomp.pairwise_tukeyhsd(endog=df_name[col_response]\
                    , groups=df_name[col_predictor], alpha=alpha)
    print(tukey_oneway.summary())
    
def lr_ols_model(df_name, col_response:str, col_cont_predictors:list, col_cat_predictors:list):
    # col_cont_predictors: list of continuous predictor variables used
    # col_cat_predictors: list of categorical predictor variables used
    # col_response: single response variable to be regressed
    # output to dict with keys: Summary, Residuals and FittedValues in that order
    addition_string = ""
    for idx in range(len(col_cont_predictors)):
        addition_string += col_cont_predictors[idx]
        if len(col_cat_predictors) > 0:
            addition_string += " + "
        elif idx < len(col_cont_predictors)-1:
            addition_string += " + "
    for idx in range(len(col_cat_predictors)):
        addition_string += "C("+col_cat_predictors[idx]+")"
        if idx < len(col_cat_predictors)-1:
            addition_string += " + "
    lr_ols_formula = "{y} ~ {x}".format(y=col_response, x=addition_string)
    lr_ols_data = df_name
    lr_OLS = statsmodels.formula.api.ols(formula=lr_ols_formula, data=lr_ols_data)
    lr_ols_model = lr_OLS.fit()
    lr_ols_model_summary = lr_ols_model.summary()
    lr_ols_model_residuals = lr_ols_model.resid
    lr_ols_model_fittedvalues = lr_ols_model.fittedvalues
    lr_output_dict = {}
    lr_output_dict['Summary'] = lr_ols_model_summary
    lr_output_dict['Residuals'] = lr_ols_model_residuals
    lr_output_dict['FittedValues'] = lr_ols_model_fittedvalues
    if (len(col_cat_predictors) + len(col_cont_predictors)) > 1: 
        print("running multiple linear regression model...")
    else:
        print("running simple linear regression model...")
    print("regressed variable: {}".format(col_response))
    print("continuous predictors: {}".format(col_cont_predictors))
    print("categorical predictors: {}".format(col_cat_predictors))
    return lr_output_dict

def logr_predictor(df_name, log_regression_model: dict):
    variables = log_regression_model['Variables']
    model = log_regression_model['Model']
    classifier = skllinmod.LogisticRegression()\
        .fit(model[0],model[2]) # Using X_train and y_train
    y_predicted = classifier.predict(model[1]) # using X_test
    logit_predicted = classifier.predict_proba(model[1])[::,-1]
    # confusion matrix
    confusion_matrix = sklmtrcs.confusion_matrix(model[3], y_predicted,\
                        labels=classifier.classes_)
    # evaluation metrics
    false_negatives = confusion_matrix[1][0]
    true_negatives = confusion_matrix[0][0]
    false_positives = confusion_matrix[0][1]
    true_positives = confusion_matrix[1][1]
    # cm = tensor[i][j]
    model_recall = true_positives/\
          (false_negatives+true_positives)
    model_precision = true_positives/\
          (false_positives+true_positives)
    model_accuracy = (true_positives+true_negatives)/\
          (false_negatives+true_negatives+\
           false_positives+true_positives)
    cm_display = sklmtrcs.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,\
                        display_labels=classifier.classes_)
    cm_display.plot()
    matplotlib.pyplot.show()
    return {"Precision": format(model_precision, "0.1%"), 
            "Recall": format(model_recall, "0.1%"), 
            "Accuracy": format(model_accuracy, "0.1%")}

def logr_classifier(df_name, log_regression_model: dict):
    variables = log_regression_model['Variables']
    model = log_regression_model['Model']
    classifier = skllinmod.LogisticRegression()\
        .fit(model[0],model[2])
    coefficient = classifier.coef_[0][0]
    intercept = classifier.intercept_[0]
    print("classifier coef: {}".format(coefficient))
    print("classifier intcpt: {}".format(intercept))
    return coefficient, intercept

def logr_train_test_split(df_name, col_response, col_predictor, test_size:float):
    X = df_name[[col_predictor]]
    y = df_name[[col_response]]
    X_train, X_test, y_train, y_test = sklmodslct.train_test_split(X,y,\
                                  test_size=test_size/100,random_state=42)
    logistic_regression = {}
    logistic_regression["Variables"] = [col_predictor, col_response]
    logistic_regression["Model"] = X_train, X_test, y_train, y_test
    return logistic_regression

def df_one_hot_enconding(df_name, col_name, *binary_bool:bool):
    df_unique_groups = df_explore_unique_categories(df_name, col_name).values.tolist()
    matching_dict = {}
    if binary_bool:
        neg = ["dis", "not", "un"]
        for each_cat in df_unique_groups:
            while isinstance(each_cat, str):
                if any(negation in each_cat for negation in neg):
                    indexer = 0
                else:
                    indexer = 1
                matching_dict[each_cat] = indexer
                break
    else:
        indexer = -1
        for each_cat in df_unique_groups:
            indexer += 1
            matching_dict[each_cat] = indexer 
    df_out = df_name
    for keys, values in matching_dict.items():
        df_out[col_name] = df_out[col_name].replace(keys, values) 
    return df_out

def df_info_dtypes(df_name):
    print(df_name.info())