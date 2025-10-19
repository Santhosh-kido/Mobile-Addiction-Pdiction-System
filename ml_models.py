from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def get_prediction_label(prediction):
    labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    return labels.get(prediction, "Low Risk")

def run_ml_algorithms(X_train, y_train, X_test, y_test, user_features):
    results = []
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_user_pred = dt.predict(user_features)[0]
    results.append({
        "algorithm": "Decision Tree",
        "prediction": get_prediction_label(dt_user_pred),
        "accuracy": round(dt_accuracy * 100),
        "addictionPercentage": round(dt_user_pred * 50)
    })
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_user_pred = rf.predict(user_features)[0]
    results.append({
        "algorithm": "Random Forest",
        "prediction": get_prediction_label(rf_user_pred),
        "accuracy": round(rf_accuracy * 100),
        "addictionPercentage": round(rf_user_pred * 50)
    })
    
    # SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_user_pred = svm.predict(user_features)[0]
    results.append({
        "algorithm": "SVM",
        "prediction": get_prediction_label(svm_user_pred),
        "accuracy": round(svm_accuracy * 100),
        "addictionPercentage": round(svm_user_pred * 50)
    })
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_user_pred = lr.predict(user_features)[0]
    results.append({
        "algorithm": "Logistic Regression",
        "prediction": get_prediction_label(lr_user_pred),
        "accuracy": round(lr_accuracy * 100),
        "addictionPercentage": round(lr_user_pred * 50)
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_user_pred = nb.predict(user_features)[0]
    results.append({
        "algorithm": "Naive Bayes",
        "prediction": get_prediction_label(nb_user_pred),
        "accuracy": round(nb_accuracy * 100),
        "addictionPercentage": round(nb_user_pred * 50)
    })
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_user_pred = gb.predict(user_features)[0]
    
    results.append({
        "algorithm": "Gradient Boosting",
        "prediction": get_prediction_label(gb_user_pred),
        "accuracy": round(gb_accuracy * 100),
        "addictionPercentage": round(gb_user_pred * 50)
    })
    
    return results

