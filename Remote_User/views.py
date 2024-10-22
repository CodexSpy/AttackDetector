from django.shortcuts import render, redirect, get_object_or_404
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from Remote_User.models import ClientRegister_Model, false_data_injection_attack_detection, detection_ratio, detection_accuracy

def login(request):
    if request.method == "POST" and 'submit1' in request.POST:
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username, password=password)
            request.session["userid"] = enter.id
            return redirect('ViewYourProfile')
        except ClientRegister_Model.DoesNotExist:
            pass  # Handle the case where login credentials are incorrect
    return render(request, 'RUser/login.html')

def Add_DataSet_Details(request):
    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})

def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request, 'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session.get('userid')
    if userid:
        obj = ClientRegister_Model.objects.get(id=userid)
        return render(request, 'RUser/ViewYourProfile.html', {'object': obj})
    else:
        return redirect('login')

def Predict_false_data_injection_attack_detection(request):
    if request.method == "POST":
        URLs = request.POST.get('URLs')
        Headline = request.POST.get('Headline')
        Body = request.POST.get('Body')

        data = pd.read_csv("Datasets.csv", encoding='latin-1')

        def apply_results(label):
            return 0 if label == 0 else 1

        data['Results'] = data['Label'].apply(apply_results)
        x = data['Body'].apply(str)
        y = data['Results']

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
        x = cv.fit_transform(x)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

        models = []

        # Naive Bayes
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        models.append(('naive_bayes', NB))

        # SVM
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        models.append(('svm', lin_clf))

        # Logistic Regression
        reg = LogisticRegression(random_state=0, solver='liblinear')
        reg.fit(X_train, y_train)
        y_pred_lr = reg.predict(X_test)
        models.append(('logistic', reg))

        # Decision Tree Classifier
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        models.append(('DecisionTreeClassifier', dtc))

        # Random Forest Classifier
        RFC = RandomForestClassifier(random_state=0)
        RFC.fit(X_train, y_train)
        pred_rfc = RFC.predict(X_test)
        models.append(('RFC', RFC))

        # Voting Classifier
        classifier = VotingClassifier(estimators=models, voting='hard')
        classifier.fit(X_train, y_train)

        Body1 = [Body]
        vector1 = cv.transform(Body1).toarray()
        predict_text = classifier.predict(vector1)

        pred = int(predict_text[0])

        val = 'No False Data Injection Attack Detected' if pred == 0 else 'False Data Injection Attack Detected'

        false_data_injection_attack_detection.objects.create(
            URLs=URLs,
            Headline=Headline,
            Body=Body,
            Prediction=val)

        return render(request, 'RUser/Predict_false_data_injection_attack_detection.html', {'objs': val})

    return render(request, 'RUser/Predict_false_data_injection_attack_detection.html')
