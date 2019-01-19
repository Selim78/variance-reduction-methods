import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, add_dummy_feature
import warnings

def load_student_data(file, split=0.25):
    """
    Load the student performance dataset (http://archive.ics.uci.edu/ml/datasets/Student+Performance) from a csv file.
    Returns preprocessed training and testing data: X_train, X_test, y_train, y_test
    """
    student  = pd.read_csv(file)
    target   = pd.DataFrame(student["G3"])
    features = student.drop(["G3"],axis=1)
    
    # Creating the target data: grade above 12 is a pass (1), under 12 it's a fail (-1)
    target = target.applymap(lambda grade: [-1, 1][grade >= 12])

    # Encoding categorical values with numerical labels
    numerical_features = features.apply(LabelEncoder().fit_transform)

    # Normalisation 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalised_features = add_dummy_feature(StandardScaler().fit_transform(numerical_features));
    preprocessed_features = pd.DataFrame(normalised_features , columns=[ "intercept" ] + list(numerical_features.columns) )

    # Train test split
    try:
        from sklearn.model_selection import train_test_split    # sklearn > ...
    except:
        from sklearn.cross_validation import train_test_split   # sklearn < ...
    
    # return X_train, X_test, y_train, y_test
    return train_test_split(np.array(preprocessed_features), np.ravel(target), test_size = split)