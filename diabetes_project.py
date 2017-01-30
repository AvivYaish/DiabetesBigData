"""
Name: Aviv Yaish
Filename: diabetes_project.py
Description: toying around with diabetic data from
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
Usage: python diabetes_project.py
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

from scipy.stats.distributions import chi2
from scipy.stats import chisquare

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import pandas as pd
import numpy as np
import warnings

# A delimiter for the various text prints
TEXT_DELIMITER = "========================================="

# A list of all the drugs
DRUG_LIST = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
             'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
             'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
             'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
             'metformin-pioglitazone']


def print_header(header):
    """
    Prints a header with the given text.
    :param header: the header to print.
    """
    print("\n\n" + TEXT_DELIMITER + "\n" + header + "\n" + TEXT_DELIMITER)


def pretty_graph(data, column, title, xlabel, ylabel, hue=None):
    """
    Plots a pretty graph.
    :param data: the data to visualize.
    :param column: either the column of the dataframe to visualize, or None if data is a value counts object.
    :param title: the title of the graph.
    :param xlabel: the xlabel of the graph.
    :param ylabel: the ylabel of the graph.
    :param hue: the column name for the hue.
    :return: the axes of the graph.
    """
    plt.figure()
    ax = plt.axes()

    if column is not None:
        if hue is None:
            sns.countplot(x=column, data=data)
            total = len(data)
        else:
            sns.countplot(x=column, hue=hue, data=data)
    else:
        sns.barplot(x=data.index, y=data)
        total = sum(data)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if hue is None:
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100 * p.get_height() / total), (x.mean(), y), ha='center', va='bottom')

    plt.tight_layout()
    return ax


def pretty_scatter(data, fit, title):
    """
    Plots a pretty scatter plot.
    :param data: a numerical matrix.
    :param fit: a KMeans fit for the data.
    :param title: the title of the plot.
    :return: the axes of the graph.
    """
    normalizer = Normalizer()
    pca = PCA(n_components=2)

    reduced_data = pca.fit_transform(normalizer.fit_transform(data))
    reduced_centroids = pca.transform(normalizer.transform(fit.cluster_centers_))

    label_num = len(np.unique(fit.labels_))
    colors = cm.rainbow(np.linspace(0, 1, label_num))
    label_colors = [colors[label] for label in fit.labels_]

    plt.figure()
    ax = plt.axes()

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=label_colors)
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], marker='x', s=169, linewidths=3, c=colors, zorder=8)

    ax.set_title(title + "\n(centroids marked with an X)")
    plt.tight_layout()

    return ax


def part_1_1_helper(diabetic_data, n=5, gender="All"):
    """
    A helper function for part 1.1.
    :param diabetic_data: the diabetic data.
    :param n: number of top ICD-9 codes to print.
    :param gender: the gender to answer gor.
    """
    if gender != "All":
        diabetic_data = diabetic_data[diabetic_data['gender'] == gender]

    pretty_graph(diabetic_data, 'age', 'Age distribution for ' + gender, 'Age', 'Count')
    pretty_graph(diabetic_data, 'race', 'Race distribution for ' + gender, 'Race', 'Count')
    pretty_graph(diabetic_data, 'readmitted', 'Readmission distribution for ' + gender, 'Readmission type', 'Count')
    top_icd9_codes = pd.concat([diabetic_data['diag_1'], diabetic_data['diag_2'], diabetic_data['diag_3']]). \
        value_counts().head(n)
    pretty_graph(top_icd9_codes, None, 'Top ICD-9 codes for ' + gender, 'ICD-9 code', 'Count')

    print("The top ICD-9 codes for " + gender + " are:")
    print(pd.concat([diabetic_data['diag_1'], diabetic_data['diag_2'], diabetic_data['diag_3']]).
          value_counts().head(n).index.tolist())
    print(TEXT_DELIMITER)


def part_1_1(diabetic_data):
    """
    The answer for part 1.1.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 1.1")

    pretty_graph(diabetic_data, 'age', 'Age distribution', 'Age', 'Count', 'gender')
    pretty_graph(diabetic_data, 'race', 'Race distribution', 'Race', 'Count', 'gender')
    pretty_graph(diabetic_data, 'readmitted', 'Readmission distribution', 'Readmission type', 'Count', 'gender')

    # More graphs
    # Go over all possible genders when answering part 1.1
    part_1_1_helper(diabetic_data, gender="All")
    for cur_gender in diabetic_data['gender'].unique():
        part_1_1_helper(diabetic_data, gender=cur_gender)


def part_1_2_helper(diabetic_data, gender="All", age="All"):
    """
    A helper function for part 1.2.
    :param diabetic_data: the diabetic data.
    :param gender: the gender to answer gor.
    :param age: the age to answer for.
    """
    if gender != "All":
        diabetic_data = diabetic_data[diabetic_data['gender'] == gender]

    if age != "All":
        diabetic_data = diabetic_data[diabetic_data['age'] == age]

    # Time in the hospital according to gender and age
    pretty_graph(diabetic_data, 'time_in_hospital', 'Days in hospital distribution for gender ' +
                 gender + " of ages " + age, 'Days in hospital', 'Count')

    # A1C test result according to gender and age
    pretty_graph(diabetic_data, 'A1Cresult', 'A1c test results distribution for gender ' +
                 gender + " of ages " + age, 'A1c test result', 'Count')


def part_1_2(diabetic_data):
    """
    The answer for part 1.2.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 1.2")

    # Time in the hospital according to gender and age
    pretty_graph(diabetic_data, 'time_in_hospital', 'Days in hospital distribution for gender',
                 'Days in hospital', 'Count', 'gender')
    pretty_graph(diabetic_data, 'time_in_hospital', 'Days in hospital distribution for age',
                 'Days in hospital', 'Count', 'age')

    # A1C test result according to gender and age
    pretty_graph(diabetic_data, 'A1Cresult', 'A1c test results distribution for gender'
                 , 'A1c test result', 'Count', 'gender')
    pretty_graph(diabetic_data, 'A1Cresult', 'A1c test results distribution for age'
                 , 'A1c test result', 'Count', 'age')

    # More graphs
    # Go over all possible genders and ages when answering part 1.1
    part_1_2_helper(diabetic_data, gender="All")
    for cur_gender in diabetic_data['gender'].unique():
        part_1_2_helper(diabetic_data, gender=cur_gender)
    for cur_age in diabetic_data['age'].unique():
        part_1_2_helper(diabetic_data, age=cur_age)


def part_1(diabetic_data):
    """
    The answer for part 1.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 1")

    part_1_1(diabetic_data)
    part_1_2(diabetic_data)


def hypothesis_test(readmission_no_treatment, readmission_treatment, treatment, alpha=0.05):
    """
    Performs an hypothesis test to check if a given treatment changes readmission distributions.
    :param readmission_no_treatment: value counts for the distribution when there is no treatment.
    :param readmission_treatment: value counts for the distribution when there is a treatment.
    :param treatment: the name of the treatment.
    :param alpha: the alpha of the test.
    """
    # I've chosen chi-squared as the hypothesis test

    print_header("Hypothesis testing for the treatment: %r\n"
                 "We will perform the chi-squared test with alpha = %r" % (treatment, alpha))

    no_treatment_readmission_percentages = readmission_no_treatment / sum(readmission_no_treatment)
    expected_readmission_treatment = sum(readmission_treatment) * no_treatment_readmission_percentages

    # check if the expected frequencies are large enough for the chi-squared test
    can_use_chi_squared = sum(expected_readmission_treatment > 5) == len(expected_readmission_treatment)
    print("Are the expected frequencies enough for the chi-squared test? %r" % can_use_chi_squared)
    if not can_use_chi_squared:
        print("Frequencies not high enough for this treatment.")
        return

    # the decision rule
    critical_value = chi2.isf(q=alpha, df=len(expected_readmission_treatment) - 1)
    print("The critical value is: %r" % critical_value)

    # perform the test
    chisq, p_value = chisquare(f_obs=readmission_treatment, f_exp=expected_readmission_treatment)
    rejection = (chisq > critical_value) and (p_value < alpha)
    print("The chi-squared value is %r, the p-value is %r.\nShould we reject H0? %r" % (chisq, p_value, rejection))
    print("Note: H0 is the hypothesis that the treatment makes no change to the readmission distribution.")

    # Graphs and data for part 1.3
    # Readmission counts for patients who had no treatment
    pretty_graph(readmission_no_treatment, None, 'Readmission count, without treatment: ' + treatment,
                 'Readmission', 'Count')

    # Readmission counts for patients who had the treatment
    pretty_graph(readmission_treatment, None, 'Readmission count, with treatment: ' + treatment,
                 'Readmission', 'Count')

    # Various values relevant
    # Note that of course that the first two percentages should be the same
    print("Some supporting statistics:")
    print("Readmission percentages without treatment:\n%r" % no_treatment_readmission_percentages)
    print("Expected readmission percentages with treatment (should be the same as above):\n%r" %
          (expected_readmission_treatment / sum(expected_readmission_treatment)))
    print("Actual readmission percentages with treatment:\n%r" % (readmission_treatment / sum(readmission_treatment)))


def part_2_1(diabetic_data):
    """
    The answer for part 2.1.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 2.1")

    readmission_no_test = diabetic_data[diabetic_data['A1Cresult'] == 'None']['readmitted'].value_counts()
    readmission_with_test = diabetic_data[diabetic_data['A1Cresult'] != 'None']['readmitted'].value_counts()
    hypothesis_test(readmission_no_test, readmission_with_test, "A1C test")


def get_prescribed_or_increased_list(diabetic_data):
    """
    :param diabetic_data: the diabetic data.
    :return: a list such that for each patient, if he was prescribed medicine or had an
    increased dosage of a medicine, there is a True in his index. Else, there is a False.
    """
    # Hypothesis testing for an increased dose for any of the drugs or a drug was prescribed
    # np.any is an or between lists, this way we can check if there was an increased dosage for at least one drug
    increased_dosage_list = np.any([np.array(diabetic_data[drug] == 'Up') for drug in DRUG_LIST], axis=0)
    medication_prescribed_list = np.array(diabetic_data['diabetesMed'] == 'No')
    # here we check if there is a drug prescribed or there is an increased dosage for at least one drug
    medication_or_increased_dosage_list = np.any([medication_prescribed_list, increased_dosage_list], axis=0)

    return medication_or_increased_dosage_list


def part_2_2(diabetic_data):
    """
    The answer for part 2.2.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 2.2")

    # Hypothesis testing for an increased dose for any of the drugs or a drug was prescribed
    medication_or_increased_dosage_list = get_prescribed_or_increased_list(diabetic_data)
    readmission_no_persription_ano_no_dosage = \
        diabetic_data[np.logical_not(medication_or_increased_dosage_list)]['readmitted'].value_counts()
    readmission_persription_or_dosage = diabetic_data[medication_or_increased_dosage_list]['readmitted'].value_counts()
    hypothesis_test(readmission_no_persription_ano_no_dosage, readmission_persription_or_dosage,
                    "drugs prescribed or increased dosage in at least one drug")


def part_2_3(diabetic_data):
    """
    The answer for part 2.3. Note that there is not such part,
    I have made more hypothesis tests to study the data deeper.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 2.3")

    # Hypothesis testing on the fact that a drug was prescribed
    # Note that this wasn't required
    readmission_no_drugs = diabetic_data[diabetic_data['diabetesMed'] == 'No']['readmitted'].value_counts()
    readmission_drugs = diabetic_data[diabetic_data['diabetesMed'] != 'No']['readmitted'].value_counts()
    hypothesis_test(readmission_no_drugs, readmission_drugs, "drugs prescribed")

    # Hypothesis testing for an increased dose for each of the drugs
    # Note that this wasn't required
    for drug in DRUG_LIST:
        readmission_same_dosage = diabetic_data[diabetic_data[drug] != 'Up']['readmitted'].value_counts()
        readmission_increased_dosage = diabetic_data[diabetic_data[drug] == 'Up']['readmitted'].value_counts()
        hypothesis_test(readmission_same_dosage, readmission_increased_dosage, "increased dosage of " + drug)

    # Hypothesis testing on the fact that there was a change in at least one drug
    # Note that this wasn't required
    readmission_no_change = diabetic_data[diabetic_data['change'] != 'Ch']['readmitted'].value_counts()
    readmission_change = diabetic_data[diabetic_data['change'] == 'Ch']['readmitted'].value_counts()
    hypothesis_test(readmission_no_change, readmission_change, "change in diabetic medications")


def binarize_readmission(diabetic_data):
    """
    :param diabetic_data: the diabetic data.
    :return: binarizes the readmission column of the diabetic data
    ('NO' and '>30' are unified into a single value).
    """
    binarized_data = diabetic_data.copy()
    binarized_data['readmitted'] = binarized_data['readmitted'].replace(["NO", ">30"], ["NO || >30", "NO || >30"])
    return binarized_data


def part_2(diabetic_data):
    """
    The answer for part 2.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 2")

    # Note that this part is in essence a feature selection method:
    # if we find that a certain feature (A1c result, drug prescription or increased dosage)
    # affects the distribution of readmission, then we know this feature will help us in
    # predicting (and clustering) patients according to readmission.

    binarized_readmission_data = binarize_readmission(diabetic_data)
    part_2_1(binarized_readmission_data)
    part_2_2(binarized_readmission_data)
    part_2_3(binarized_readmission_data)


def gen_diagnostic_columns(diabetic_data, min_frequency=500):
    """
    :param diabetic_data: the diabetic data.
    :param min_frequency: the minimal frequency of a diagnosis that should be included.
    :return: a DataFrame with one column for each diagnosis (which occurred more than min_frequency times),
    and one row for each patient, where if a patient was diagnosed with diagnosis X, he will have a 1 in the X column.
    Otherwise, he will have a 0.
    """
    specific_diagnostic_columns_dict = {}

    # get all possible diagnoses
    all_diags = set(diabetic_data['diag_1'].unique()) | set(diabetic_data['diag_2'].unique()) | \
                set(diabetic_data['diag_3'].unique())

    diagnostic_columns = ['diag_1', 'diag_2', 'diag_3']

    for diag in all_diags:
        all_patients_with_diag = np.any([diabetic_data[column] == diag for column in diagnostic_columns], axis=0)
        if sum(all_patients_with_diag) > min_frequency:
            specific_diagnostic_columns_dict[diag] = all_patients_with_diag.astype(int)

    return pd.DataFrame(specific_diagnostic_columns_dict)


def part_3_1_initial(diabetic_data, part=3):
    """
    An implementation for part 3.1, creates a one hot representation for 'readmitted' and 'diabetesMed',
    and a 0/1 representation for the diagnoses.
    :param diabetic_data: the diabetic data.
    :param part: if used in part 3 or 4.
    :return: the title of the representation, the vectorizer for the representation,
             and a vector representation for the data.
    """
    title = "initial vectorizer"

    # Note: it is very easy to add more features to the one-hot representation, simply
    # add the column name to this array
    column_names = ['diabetesMed', 'readmitted']

    if part == 3:
        print_header("Part 3.1: " + title)
        print("An implementation for part 3.1, creates a one hot representation for 'diabetesMed', "
              "'readmitted', and a 0/1 representation for the diagnoses.")
    else:
        column_names = ['diabetesMed']

    # process the data
    diagnostic_columns = gen_diagnostic_columns(diabetic_data)
    selected_data = pd.concat([diabetic_data[column_names], diagnostic_columns], axis=1)

    # vectorize the data to allow clustering
    vectorizer = DictVectorizer(sparse=False)
    vector_representation = vectorizer.fit_transform(selected_data.to_dict(orient='records'))

    return title, vectorizer, vector_representation


def part_3_1_more_features(diabetic_data, part=3):
    """
    An implementation for part 3.1 with more features. Creates a one hot representation for 'A1Cresult',
    'diabetesMed', 'readmitted', and a 0/1 representation for the diagnoses and for (medicine prescribed or
    increased dosage).
    :param diabetic_data: the diabetic data.
    :param part: if used in part 3 or 4.
    :return: the title of the representation, the vectorizer for the representation,
    and a vector representation for the data.
    """
    title = "more features vectorizer"

    # Note: it is very easy to add more features to the one-hot representation, simply
    # add the column name to this array
    column_names = ['A1Cresult', 'readmitted']

    if part == 3:
        print_header("Part 3.1: " + title)
        print("An implementation for part 3.1 with more features. "
              "Creates a one hot representation for 'A1Cresult', 'readmitted', "
              "and a 0/1 representation for the diagnoses and for (medicine prescribed or increased dosage).")
    else:
        column_names = ['A1Cresult']

    # process the data
    diagnostic_columns = gen_diagnostic_columns(diabetic_data)
    medicine_prescribed_or_increased_dosage = pd.DataFrame(get_prescribed_or_increased_list(diabetic_data).astype(int),
                                                           columns=["medicine prescribed or increased dosage"])
    selected_data = pd.concat([diabetic_data[column_names], medicine_prescribed_or_increased_dosage,
                               diagnostic_columns], axis=1)

    # vectorize the data to allow clustering
    vectorizer = DictVectorizer(sparse=False)
    vector_representation = vectorizer.fit_transform(selected_data.to_dict(orient='records'))

    return title, vectorizer, vector_representation


def part_3_2(vectorizer, vectorized_data, cluster_num, title):
    """
    The answer for part 3.1.
    :param vectorizer: the vectorizer for the representation.
    :param vectorized_data: a vector representation for the data.
    :param cluster_num: the number of clusters to use.
    :param title: the title of the vectorizer.
    :return: the score of the KMeans.
    """
    print_header("Part 3.2: " + title)

    km = KMeans(n_clusters=cluster_num)
    fit = km.fit(vectorized_data)
    score = km.score(vectorized_data)

    # some data for part 1.3
    print("The score for %r based clustering on the data is %r." % (title, score))

    print("There are %r clusters. The centroids are:" % cluster_num)
    for cur_num, centroid in enumerate(fit.cluster_centers_, start=0):
        print("Centroid number %r: %r" % (cur_num, vectorizer.inverse_transform(np.round(centroid))))
    print("Note: no values (for example, for drugs) means that the centroid had a 0 there.\n")

    label_counts = pd.Series(fit.labels_).value_counts().sort_index()
    print("The percentage of patients allocated to each centroid is:\n%r" % (100 * label_counts / sum(label_counts)))

    pretty_scatter(vectorized_data, fit, title + "\n" + str(cluster_num) + " clusters")

    return score


def part_3(diabetic_data):
    """
    The answer for part 3.
    :param diabetic_data: the diabetic data.
    """
    print_header("Part 3")

    # Note that given two points, x and x', K Means regards similarity(x, x')
    # (or metric(x, x')) as the l2 difference between them: ||x - x'||.
    # So, by defining a vector representation of the data, we are also defining
    # the similarity (metric) for the data.
    # I will explore different vector representations and answer part 3.2 on each:

    representations = [part_3_1_initial, part_3_1_more_features]
    for representation in representations:
        title, vectorizer, vectorized_data = representation(diabetic_data)
        print("The features' names for %r are:\n%r" % (title, vectorizer.get_feature_names()))
        part_3_2(vectorizer, vectorized_data, 5, title)
        part_3_2(vectorizer, vectorized_data, 10, title)


def generate_x_y(data):
    """
    :param data: the diabetic data to separate.
    :return: a separation of the data into X and y (labels).
    """
    X = data.drop('readmitted', 1)
    y = data['readmitted'].replace(['NO', '>30', '<30'], [0, 1, 2])
    binary_y = binarize_readmission(data)['readmitted'].replace(['NO || >30', '<30'], [0, 1])
    return X, y, binary_y


def print_important_features(clf, x, vectorizer, title=''):
    """
    Prints the feature importances for the given classifier.
    :param clf: a tree based classifier.
    :param x: the training data the classifier was fitted on.
    :param title: title for the various prints.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = vectorizer.get_feature_names()

    # Print the feature ranking
    print("Feature ranking for %r:" % title)
    for f in range(x.shape[1]):
        print("%d. feature %r (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances
    plt.figure()
    ax = plt.axes()

    ax.set_title("Feature importances for " + title)
    ax.bar(range(x.shape[1]), importances[indices], color="r", align="center")
    ax.set_xticks(range(x.shape[1]))
    ax.set_xlim([-1, x.shape[1]])

    plt.tight_layout()


def part_4_helper(x_train, ternary_y_train, binary_y_train, x_valid,
                  ternary_y_valid, binary_y_valid, clf, split, do_after,
                  title, vectorizer):
    """
    A helper function for part 4.
    :param x_train: X data for training.
    :param ternary_y_train: ternary labels for training.
    :param binary_y_train: binary labels for training.
    :param x_valid: X data for validation.
    :param ternary_y_valid: ternary labels for validation.
    :param binary_y_valid: binary labels for validation.
    :param clf: the classifier to use.
    :param split: the training data split to use.
    :param do_after: a function to run after training.
    :param title: title for the various prints.
    :param vectorizer: the vectorizer that generated the x-es.
    """
    binary_results = []
    ternary_results = []
    confusion_matrices = []

    for train_index, test_index in split:
        # split data
        cross_x_train, cross_x_test = x_train[train_index], x_train[test_index]
        cross_binary_y_train, cross_binary_y_test = binary_y_train[train_index], binary_y_train[test_index]
        cross_ternary_y_train, cross_ternary_y_test = ternary_y_train[train_index], ternary_y_train[test_index]

        # binary classification
        clf.fit(cross_x_train, cross_binary_y_train)
        binary_results.append(clf.score(cross_x_test, cross_binary_y_test))

        # ternary classification
        clf.fit(cross_x_train, cross_ternary_y_train)
        prediction = clf.predict(cross_x_test)
        ternary_results.append(np.mean(prediction != cross_ternary_y_test))
        confusion_matrices.append(confusion_matrix(cross_ternary_y_test, prediction))

    print("The mean test error of the binary classification is: %r." % np.mean(binary_results))
    print("The median test error of the binary classification is: %r." % np.median(binary_results))
    print("The mean test error of the ternary classification is: %r." % np.mean(ternary_results))
    print("The median test error of the ternary classification is: %r." % np.median(ternary_results))
    print("The mean test confusion matrix of the ternary classification is:\n%r." %
          np.mean(confusion_matrices, axis=0))
    print("The median test confusion matrix of the ternary classification is:\n%r." %
          np.median(confusion_matrices, axis=0))

    # validate model after training on all the training data
    clf.fit(x_train, binary_y_train)
    print("The validation error of the binary classification is: %r." % clf.score(x_valid, binary_y_valid))
    if do_after:
        do_after(clf, x_train, vectorizer, title + " (binary classification)")

    clf.fit(x_train, ternary_y_train)
    prediction = clf.predict(x_valid)
    print("The validation error of the ternary classification is: %r." % np.mean(prediction != ternary_y_valid))
    print("The validation confusion matrix of the ternary classification is:\n%r." %
          confusion_matrix(ternary_y_valid, prediction))
    if do_after:
        do_after(clf, x_train, vectorizer, title + " (ternary classification)")


def part_4(train_data, test_data):
    """
    The answer for part 4.
    :param train_data: the train data.
    :param test_data: the test data.
    """
    print_header("Part 4")

    # separate data
    x_train, y_train, binary_y_train = generate_x_y(train_data)
    x_test, y_test, binary_y_test = generate_x_y(test_data)

    # note: all classifiers run on the same split, for comparison purposes
    kf = KFold(n_splits=9)
    split = list(kf.split(x_train))

    # we will compare all classifiers under all representations
    representations = [part_3_1_initial, part_3_1_more_features]
    classifiers = [
        ("Gaussian Naive Bayes", GaussianNB(), None),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis(), None),
        ("Multi-Layer Perceptron, alpha=1", MLPClassifier(alpha=1), None),
        ("Linear Support Vector Classifier", LinearSVC(), None),
        ("Decision Tree", DecisionTreeClassifier(), print_important_features),
        ("Random Forest", RandomForestClassifier(), print_important_features),
        ("KNN, k=1", KNeighborsClassifier(n_neighbors=1), None),
        ("KNN, k=2", KNeighborsClassifier(n_neighbors=2), None),
        ("KNN, k=3", KNeighborsClassifier(n_neighbors=3), None),
        ("KNN, k=4", KNeighborsClassifier(n_neighbors=4), None),

    ]

    for representation in representations:
        # process X
        representation_title, vectorizer, x_train_vec = representation(x_train, 4)
        x_test_vec = vectorizer.transform(x_test.to_dict(orient='records'))

        # classify
        for classifier_title, classifier, do_after in classifiers:
            title = classifier_title + " with representation: " + representation_title
            print_header(title)
            part_4_helper(x_train_vec, y_train, binary_y_train,
                          x_test_vec, y_test, binary_y_test, classifier,
                          split, do_after, title, vectorizer)


def main():
    """
    Loads the data and runs all different parts.
    """
    # Load data
    diabetic_data = pd.read_csv("diabetic_data.csv")
    train_data, test_data = train_test_split(diabetic_data, test_size=0.1)

    reindexed_train_data = train_data.copy().reset_index()
    reindexed_test_data = test_data.reset_index()

    # Parts
    # Part 1 - show general statistics about the data.
    part_1(train_data)

    # Part 2 - perform hypothesis tests to find correlation between readmission and the other data.
    part_2(train_data)

    # Part 3 - define a metric on the data and cluster it.
    part_3(reindexed_train_data.copy())

    # Part 4 - try to use machine learning to predict readmission.
    part_4(reindexed_train_data, reindexed_test_data)

    plt.show()
    return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
