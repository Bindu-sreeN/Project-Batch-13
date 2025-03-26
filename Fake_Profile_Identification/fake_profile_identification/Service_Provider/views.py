# Import necessary modules from Django
from django.db.models import Count, Avg  # To perform aggregate functions on database queries
from django.shortcuts import render, redirect  # To render templates and handle redirects
from django.db.models import Q  # To filter database queries using OR conditions
from django.http import HttpResponse  # To send HTTP responses
import datetime  # To handle date and time-related operations
import xlwt  # To generate Excel files

# Import necessary libraries for data processing and machine learning
import string
import re  # For regular expressions
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For data visualization
from sklearn.ensemble import VotingClassifier  # Ensemble learning method
from sklearn.feature_extraction.text import CountVectorizer  # To convert text to numerical data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score  # Model evaluation metrics
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
import openpyxl  # For working with Excel files

# Import models from the Remote_User application
from Remote_User.models import ClientRegister_Model, profile_identification_type, detection_ratio, detection_accuracy

# Service Provider Login View
def serviceproviderlogin(request):
    if request.method == "POST":
        admin = request.POST.get('username')  # Retrieve the username from form input
        password = request.POST.get('password')  # Retrieve the password from form input
        if admin == "Admin" and password == "Admin":  # Check if admin credentials are correct
            return redirect('View_Remote_Users')  # Redirect to the remote users' view if login is successful

    return render(request, 'SProvider/serviceproviderlogin.html')  # Render the login page

# View for displaying Profile Identity Predictions
def View_Profile_Identity_Prediction(request):
    obj = profile_identification_type.objects.all()  # Fetch all profile identification records
    return render(request, 'SProvider/View_Profile_Identity_Prediction.html', {'objs': obj})  # Pass data to the template

# View for calculating and displaying profile identity prediction ratio
def View_Profile_Identity_Prediction_Ratio(request):
    detection_ratio.objects.all().delete()  # Clear previous ratio data

    # Calculate the percentage of genuine profiles
    kword = 'Genuine Profile'
    obj = profile_identification_type.objects.filter(Prediction=kword)
    total_profiles = profile_identification_type.objects.count()
    genuine_ratio = (obj.count() / total_profiles) * 100 if total_profiles > 0 else 0
    if genuine_ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=genuine_ratio)

    # Calculate the percentage of fake profiles
    kword1 = 'Fake Profile'
    obj1 = profile_identification_type.objects.filter(Prediction=kword1)
    fake_ratio = (obj1.count() / total_profiles) * 100 if total_profiles > 0 else 0
    if fake_ratio != 0:
        detection_ratio.objects.create(names=kword1, ratio=fake_ratio)

    obj = detection_ratio.objects.all()  # Retrieve updated ratio data
    return render(request, 'SProvider/View_Profile_Identity_Prediction_Ratio.html', {'objs': obj})  # Render the template

# View for displaying registered remote users
def View_Remote_Users(request):
    obj = ClientRegister_Model.objects.all()  # Fetch all registered users
    return render(request, 'SProvider/View_Remote_Users.html', {'objects': obj})  # Render user list page

# View for generating charts based on profile detection ratios
def charts(request, chart_type):
    chart_data = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))  # Compute the average ratio
    return render(request, "SProvider/charts.html", {'form': chart_data, 'chart_type': chart_type})

# View for generating accuracy charts
def charts1(request, chart_type):
    chart_data = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts1.html", {'form': chart_data, 'chart_type': chart_type})

# View for generating likes chart based on detection accuracy
def likeschart(request, like_chart):
    chart_data = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/likeschart.html", {'form': chart_data, 'like_chart': like_chart})

# View for generating likes chart based on detection ratio
def likeschart1(request, like_chart):
    chart_data = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/likeschart1.html", {'form': chart_data, 'like_chart': like_chart})

# View for downloading the trained datasets as an Excel file
def Download_Trained_DataSets(request):
    response = HttpResponse(content_type='application/ms-excel')  # Set response type to Excel
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'  # Set filename
    wb = xlwt.Workbook(encoding='utf-8')  # Create a new workbook
    ws = wb.add_sheet("sheet1")  # Add a new sheet

    font_style = xlwt.XFStyle()  # Define font style
    font_style.font.bold = True  # Set header font to bold

    obj = profile_identification_type.objects.all()  # Fetch all profile identification data
    row_num = 0  # Initialize row counter

    # Write data to the Excel sheet
    for my_row in obj:
        row_num += 1
        ws.write(row_num, 0, my_row.prof_idno, font_style)
        ws.write(row_num, 1, my_row.name, font_style)
        ws.write(row_num, 18, my_row.Prediction, font_style)

    wb.save(response)  # Save the Excel file to response
    return response  # Return response for download

# View for training and testing datasets
def Train_Test_DataSets(request):
    detection_accuracy.objects.all().delete()  # Clear previous accuracy data

    df = pd.read_csv('Profile_Datasets.csv')  # Load dataset

    # Function to clean text data
    def clean_text(text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        return text

    df['processed_content'] = df['name'].apply(lambda x: clean_text(x))  # Apply text cleaning

    # Function to label data
    def apply_results(label):
        return 0 if label == 0 else 1

    df['results'] = df['Label'].apply(apply_results)

    cv = CountVectorizer(lowercase=False)  # Initialize CountVectorizer

    y = df['results']  # Target labels
    X = df["id"].apply(str)  # Convert IDs to string for vectorization

    X = cv.fit_transform(X)  # Convert text data into feature vectors

    # Split dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Naive Bayes Model
    from sklearn.naive_bayes import MultinomialNB
    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

    # SVM Model
    from sklearn import svm
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    # KNeighborsClassifier Model
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    knpredict = kn.predict(X_test)
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    obj = detection_accuracy.objects.all()  # Fetch accuracy results
    return render(request, 'SProvider/Train_Test_DataSets.html', {'objs': obj})  # Render results
