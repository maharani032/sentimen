# from cgitb import text
from enum import auto
import threading
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import tkinter
from tkinter.font import Font
from turtle import width
import pandas as pd
import os
import csv
import re
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import asyncio
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from PIL import ImageTk,Image
from time import time
# import time
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import tweepy
from datetime import timedelta, datetime
from dotenv import load_dotenv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg





def browseFiles():
    status.set('Browser File')
    statusLabel.update()
    filename = filedialog.askopenfilename(
        initialdir = "/",
        title = "Select a File xlsv or csv",
        filetypes = (
            ("Excel file","*.xlsx*"),
            ("CSV file","*.csv*")
            )
    ) 
    try:
        status.set('Add File')
        statusLabel.update()
        if os.path.splitext(filename)[1].lower() == '.csv':
            data=pd.read_csv(filename)
        elif os.path.splitext(filename)[1].lower() == '.xlsx':
            data= pd.read_excel(filename)
        jumlah_per_kolom = data.count()
        filePath.set(filename)
        fileName.set(os.path.basename(filename))        
        file_column.set(' '.join([f'{kolom}: {jumlah_per_kolom[kolom]} ' for kolom in jumlah_per_kolom.index]))
        new_list_column=list(data.columns)
        
        entry_c_tweet.configure(values=new_list_column,state='readonly')
        entry_c_klasifikasi.configure(values=new_list_column,state='readonly')
        entry_c_cleanData.configure(values=new_list_column,state='readonly')
        Load_excel_data(filePath=filename)
        status.set('Ready...')
        statusLabel.update()
        return None
    except ValueError:
        messagebox.showerror("Information", "The file you have chosen is no data")
        return None
    except FileNotFoundError:
        messagebox.showerror("Information", f"No such file as {fileName}")
        return None
def closeapp():
    root.destroy()
def Load_excel_data(filePath):
    try:
        excel_filename = r"{}".format(filePath)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)

    except ValueError:
        messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        messagebox.showerror("Information", f"No such file as {filePath}")
        return None

    clear_data()
    tree["column"] = list(df.columns)
    tree["show"] = "headings"
    for column in tree["columns"]:
        tree.heading(column, text=column)
        tree.column(column, stretch=False)
    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tree.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
    # scrollbar
    hs=Scrollbar(tableFrame,orient=HORIZONTAL,command=tree.xview)
    tree.configure(xscrollcommand=hs.set)
    hs.pack(side=BOTTOM,fill=X)
    vs=Scrollbar(tableFrame,orient=VERTICAL,command=tree.yview)
    tree.configure(yscrollcommand=vs.set)
    vs.pack(side=RIGHT,fill=Y)
    tree.pack(side=LEFT)

    # hs.grid(row=0,column=1,sticky='ew')
    # hs.pack(side='BOTTOM',fill=X)
    return None
def handle_click(event):
    if tree.identify_region(event.x, event.y) == "separator":
        return "break"
def clear_data():
    tree.delete(*tree.get_children())
    return None
def case_folding(tweet):
    return tweet.lower()
def remove_mention(tweet,pattern_regex):
    r=re.findall(pattern_regex,tweet)
    for i in r:
        tweet=re.sub(i,"",tweet)
    return tweet
def removeHastag(tweet):
#     hastag dolar sign
    tweet=re.sub(r'\$\w*','',tweet)
#     retweet lama
    tweet=re.sub(r'^RT[\s]+','',tweet)
#     hastag
    tweet=re.sub(r'#', '',tweet)
#     hapus angka
    tweet=re.sub('[0-9]+','',tweet)
    return tweet
def remove_http(tweet):
    tweet=" ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",tweet).split())
    return tweet
async def clean_tweets(tweet):
    stop_factory = StopWordRemoverFactory().get_stop_words()
    stopwords_indonesia=stopwords.words('indonesian')
    more_stopword = [
    'yg','dgn','utk','gw','gue','deh','gua','lu','lo','kalo','trs','jd','nih','ntar','nya','lg'
    ,'dr','kpn','kok','kyk','dong','donk','yah','tuh','si','siii','wkwk','wkwkwk','ini','mmg','jd',
    'wow','wowwwwwah','icymi','ni','coy','coii','isenkisenk','dg','pdhl','aja','tadi','krn','tak',
    'aja','sbb','kuy','se','skrg','yep','aja','as','yaa','jadinya','aja','coba','tibatiba','shit','knp','jdi','udah'
    ,'sih','bang','oke','nah','bgt','km','ttg','dlm','aaa','kang','hehe','wes','you','doang','kamu','wkkw','ong','sm','he','yeee'
    ,'emg','kak','gan','woy','dm','hi','kakk','min','di','noh','gais','lah','xfxf','nak','bro','x','ahhh','gasss','hmmm','sat','set','yukkkkk'
    ,'smh','eh','ni','laaah',' aihihi','fafifu','akwkaskaksawska '
    ]
    stopword=stop_factory+more_stopword+stopwords_indonesia
    dictionary=ArrayDictionary(stopword)
    str=StopWordRemover(dictionary)

    factory=StemmerFactory()
    stemmer=factory.create_stemmer()
#     tokenize tweets
    tokenizer=TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
    tweet_token=tokenizer.tokenize(tweet)
    
    tweet_clean=[]
    for word in tweet_token:
        if(word not in stopword):
            stem_word=stemmer.stem(word)
            tweet_clean.append(stem_word)
    return  tweet_clean
def tweet(tweet):
    return " ".join(tweet)
def preprocessing(filePath,dataTweet,dataKlasifikasi):
    print(filePath, dataTweet, dataKlasifikasi)
    # validasi jika data tweet kosong dan data klasifikasi
    if(len(dataTweet)<1):
        return messagebox.showerror("Information", "data tweet kosong")
    if(len(dataKlasifikasi)<1):
        return messagebox.showerror("Information", "data klasifikasi kosong")

    excel_filename = r"{}".format(filePath)
    if excel_filename[-4:] == ".csv":
        df = pd.read_csv(excel_filename)
    else:
        df = pd.read_excel(excel_filename)
    # Check Label
    if df[dataKlasifikasi].dtype == 'int64' or df[dataKlasifikasi].dtype == 'float64':
        # Change Label
        print("The 'label' column contains numerical data.")
        label_dict = {0: 'netral', 1: 'positif', -1: 'negatif'}
        df[dataKlasifikasi] = df[dataKlasifikasi].replace(label_dict)
    else:
        label_dict = {'Netral': 'netral', 'Positif': 'positif', 'Negatif': 'negatif'}
        df[dataKlasifikasi] = df[dataKlasifikasi].replace(label_dict)
    if (df[dataKlasifikasi].count()!= df[dataTweet].count()):
        return messagebox.showerror("Information", "panjang data tidak sesuai")
    df['remove_mention']=np.vectorize(remove_mention)(df[dataTweet]," *RT* | *@[\w]*")
    df['remove_http']=df['remove_mention'].apply(lambda x:remove_http(x))
    df['remove_hastag']=df['remove_http'].apply(lambda x:removeHastag(x))
    df['case_folding']=df['remove_hastag'].apply(lambda x:case_folding(x))

    # set time
    # update_time()
    t = time()
    status.set('Running Stemming. Please dont close...')
    statusLabel.update()
    # Stemming Processing
    df['tokenizer']= df['case_folding'].apply(lambda x:asyncio.run(clean_tweets(x)))
    print(df['tokenizer'])
    df['cleantweet']=df['tokenizer'].apply(lambda x:tweet(x))
    print(df['cleantweet'])
    # Munculkan Berapa Lama waktu yang dibutuhkan untuk process stemming
    time_spent = time() - t
    status.set('time: %0.2fs' % time_spent)
    statusLabel.update()

    df.dropna(subset=['cleantweet'], inplace=True)
    # Save File
    files = [
            ("Excel file","*.xlsx"),
            ("CSV file","*.csv")]
    file = filedialog.asksaveasfile(mode='w',filetypes = files, defaultextension = files)
    if(file is not None):
        excel_filename = r"{}".format(file.name)
        if excel_filename[-4:] == ".csv":
            df.to_csv(file.name, index=False, encoding='utf-8')
        else:
            df.to_excel(file.name, index=False, encoding='utf-8')
    status.set('Ready to Klasifikasi')
    statusLabel.update()
def naiveBayes(filePath,dataTweet,dataKlasifikasi,dataClean):
    if(len(dataTweet)<1 or len(dataClean)<1 or len(dataKlasifikasi)<1 or len(filePath)<1):
        return messagebox.showerror("Information", "data tweet kosong")
    excel_filename = r"{}".format(filePath)
    if excel_filename[-4:] == ".csv":
        df = pd.read_csv(excel_filename)
    else:
        df = pd.read_excel(excel_filename)
    if (df[dataKlasifikasi].count()!= df[dataTweet].count()):
        return messagebox.showerror("Information", "panjang data tidak sesuai")
    status.set('Running NBC please dont close')
    statusLabel.update()
    try:
        # tdf-id
        df = df.dropna(subset=[dataKlasifikasi]).reset_index(drop=True)
        # hitung term frequency (tf) dari data teks
        bow_transformer = CountVectorizer().fit(df[dataClean])
        tokens = bow_transformer.get_feature_names_out()
        text_bow = bow_transformer.transform(df[dataClean])
        # hitung tf-idf dari data teks
        tfidf_transformer=TfidfTransformer().fit(text_bow)
        tweet_tfidf=tfidf_transformer.transform(text_bow)
        # buat dataframe dari hasil tf-idf
        dd=pd.DataFrame(data=tweet_tfidf.toarray(),columns=tokens)
        # bagi dataset menjadi data train dan data test
        X = text_bow.toarray()
        Y=df[dataKlasifikasi]
        x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=35)
        # train model Naive Bayes
        model = MultinomialNB().fit(x_train,y_train)
        # membuat prediksi untuk data train dan data test
        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)

        score1 = metrics.accuracy_score(y_test, test_prediction)
        print("accuracy:   %0.3f" % score1)
        akurasi.set(score1)

        # Create popup window
        popUpNaive= Toplevel(root)
        # create frame
        figureFrame=Frame(popUpNaive,borderwidth=3)
        inputFrame=Frame(popUpNaive,borderwidth=3)
        bFrame=Frame(popUpNaive,borderwidth=3)
        for frame in [ figureFrame,inputFrame,bFrame]:
            frame.pack(padx=10,pady=10,fill='x')
            # frame.pack(fill="both", expand=True)
        popUpNaive.title("Data Naive Bayes")
        # popUpNaive.geometry("1000x400")
        my_font1=('times', 10, 'normal')
        # Data train
        x_train_df = pd.DataFrame(x_train, columns=tokens)
        train_df = pd.DataFrame({'tweet': df.iloc[x_train_df.index][dataClean],
                        'prediction': train_prediction})
        # create train table
        train_label = LabelFrame(figureFrame, text='Train Data',font=my_font1,borderwidth=3)
        train_label.pack()
        train_table = ttk.Treeview(train_label)
        train_table['columns']=('tweet','prediction')
        train_table['show']='headings'
        for column in train_table['columns']:
            train_table.heading(column, text=column)
            train_table.column(column,stretch=False)
        train_list = train_df.to_numpy().tolist()
        for row in train_list:
            train_table.insert("","end",values=row)
        # scrollbar
        hs=Scrollbar(train_label,orient=HORIZONTAL,command=train_table.xview)
        train_table.configure(xscrollcommand=hs.set)
        hs.pack(side=BOTTOM,fill='x')
        vs=Scrollbar(train_label,orient=VERTICAL,command=train_table.yview)
        train_table.configure(yscrollcommand=vs.set)
        vs.pack(side=RIGHT,fill='y')
        train_table.pack()
        # data uji 
        # x_test_df = pd.DataFrame(x_test, columns=tokens)
        # x_test_df['label'] = y_test
        # test_df = pd.DataFrame({'tweet': df.iloc[x_test_df.index][dataClean],
        #                 'prediction': test_prediction})
        # # test table
        # test_label = LabelFrame(figureFrame, text='Test Data',font=my_font1,borderwidth=3)
        # test_label.pack(side=RIGHT)
        # test_table = ttk.Treeview(test_label)
        # test_table['columns']=('tweet','prediction')
        # test_table['show']='headings'
        # for column in test_table['columns']:
        #     test_table.heading(column, text=column)
        #     test_table.column(column, width=100,stretch=False)
        # test_list = test_df.to_numpy().tolist()
        # for row in test_list:
        #     test_table.insert("","end",values=row)
        # # scrollbar
        # hs=Scrollbar(test_label,orient=HORIZONTAL,command=test_table.xview)
        # test_table.configure(xscrollcommand=hs.set)
        # hs.pack(side=BOTTOM,fill='x')
        # vs=Scrollbar(test_label,orient=VERTICAL,command=test_table.yview)
        # test_table.configure(yscrollcommand=vs.set)
        # vs.pack(side=RIGHT,fill='y')
        # test_table.pack()
        
        
        # label input text
        label_input_nbc=Label(inputFrame,text='input text:',font=my_font1)
        label_input_nbc.grid(row=1, column=0)

        # input inputnbc
        entry_nbc = Entry(inputFrame,font=my_font1,textvariable=inputnbc,width=50 )
        entry_nbc.grid(row=1, column=1 ,padx=10,sticky='w')

        # label input text
        label_result_nbc=Label(inputFrame,text='hasilnya:',font=my_font1)
        label_result_nbc.grid(row=2, column=0 ,pady=4,sticky='w')

        # input inputnbc
        entry_nbc_result = Entry(inputFrame,font=my_font1,textvariable=resultnbc,width=50 )
        entry_nbc_result.grid(row=2, column=1 ,padx=10,sticky='w')
        entry_nbc_result.config(state= "disabled")

        nbc_input_button=Button(bFrame,text='Enter',command=lambda: nbc_test(entry_nbc.get(), bow_transformer, model))
        nbc_input_button.grid(row=0,column=0)
        cm_button=Button(bFrame,text='Confusion matrix',command=lambda: showConfusionMatrixCanvas(y_test,test_prediction))
        cm_button.grid(row=1,column=0)
        cr_button=Button(bFrame,text='Classification report',command=lambda: classification_report(y_test,test_prediction))
        cr_button.grid(row=1,column=1)
        status.set('Ready...')
        statusLabel.update()
        return None
    except ValueError:
        return messagebox.showerror("Information",ValueError)
def classification_report(y_test,test_prediction):
    global classificationTop
    classificationTop=Toplevel()
    classificationTop.title('Classification Report')
    text = Text(classificationTop)
    text.pack()
    report = metrics.classification_report(y_test, test_prediction, target_names=['negatif', 'netral', 'positif'])
    text.config(width=60, height=10)
    text.insert(END, report)
def showConfusionMatrixCanvas(y_test,test_prediction):
    global confusionTop
    confusionTop=Toplevel()
    confusionTop.title('Confusion Matrix')
    columns = ['negatif','netral','positif']
    confm = confusion_matrix(y_test, test_prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm,  display_labels=columns)
    df_cm = DataFrame(confm, index=columns, columns=columns)
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(3, 3))
    ax = sn.heatmap(df_cm, cmap='Greens', annot=True)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Label prediksi')
    ax.set_ylabel('Label sebenarnya')
    # Tampilkan figure di dalam canvas tkinter
    canvas = FigureCanvasTkAgg(fig, master=confusionTop)
    canvas.get_tk_widget().pack()
def nbc_test(inputnbc, bow_transformer, model):
    if(inputnbc==""):
        return None
    test_1_unseen =  bow_transformer.transform([inputnbc])
    data=test_1_unseen.toarray()
    prediction_unseen = model.predict(data)
    return resultnbc.set(prediction_unseen)
def knn(filePath,dataTweet,dataKlasifikasi,dataClean,k):
    if(len(dataTweet)<1 or len(dataClean)<1 or len(dataKlasifikasi)<1 or len(filePath)<1 or k is None):
        return messagebox.showerror("Information", "data tweet kosong")
    excel_filename = r"{}".format(filePath)
    if excel_filename[-4:] == ".csv":
        df = pd.read_csv(excel_filename)
    else:
        df = pd.read_excel(excel_filename)
    status.set('Running KNN please dont close...')
    statusLabel.update()
    
    # tdf-id
    bow_transformer = CountVectorizer().fit(df[dataClean])
    tokens = bow_transformer.get_feature_names_out()
    text_bow = bow_transformer.transform(df[dataClean])
    tfidf_transformer=TfidfTransformer().fit(text_bow)
    tweet_tfidf=tfidf_transformer.transform(text_bow)
    dd=pd.DataFrame(data=tweet_tfidf.toarray(),columns=tokens)
    # train KNN
    X = text_bow.toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, df[dataKlasifikasi],test_size=0.2, random_state=35)
    # membuat model KNN
    knn = KNeighborsClassifier(n_neighbors=int(k))

    # # melatih model KNN dengan data training
    knn.fit(x_train, y_train)

    # # melakukan prediksi pada data testing
    y_pred = knn.predict(x_test)
    x_pred=knn.predict(x_train)

    # # menghitung performa model pada data testing
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    akurasiKNN.set(acc)
    columns = ['negatif','netral','positif']
    # Create popup window
    popUpKNN= Toplevel(root)
    #create frame
    figureFrame=Frame(popUpKNN,borderwidth=3)
    inputFrame=Frame(popUpKNN,borderwidth=3)
    bFrame=Frame(popUpKNN,borderwidth=3)
    for frame in [ figureFrame,inputFrame,bFrame]:
        frame.pack(padx=10,pady=10,fill='x')
    popUpKNN.title("Data KNN")

    my_font1=('times', 10, 'normal')
#     # Data train
    x_train_df = pd.DataFrame(x_train, columns=tokens)
    train_df = pd.DataFrame({'tweet': df.iloc[x_train_df.index][dataClean],
                    'prediction': x_pred})
#     # create train table
    train_label = LabelFrame(figureFrame, text='Train Data',font=my_font1,borderwidth=3)
    train_label.pack(side=LEFT)
    train_table = ttk.Treeview(train_label)
    train_table['columns']=('tweet','prediction')
    train_table['show']='headings'
    for column in train_table['columns']:
        train_table.heading(column, text=column)
        train_table.column(column, width=100,stretch=False)
    train_list = train_df.to_numpy().tolist()
    for row in train_list:
        train_table.insert("","end",values=row)
    # scrollbar
    hs=Scrollbar(train_label,orient=HORIZONTAL,command=train_table.xview)
    train_table.configure(xscrollcommand=hs.set)
    hs.pack(side=BOTTOM,fill='x')
    vs=Scrollbar(train_label,orient=VERTICAL,command=train_table.yview)
    train_table.configure(yscrollcommand=vs.set)
    vs.pack(side=RIGHT,fill='y')
    train_table.pack()
    # data uji 
    # x_test_df = pd.DataFrame(x_test, columns=tokens)
    # x_test_df['label'] = y_test
    # test_df = pd.DataFrame({'tweet': df.iloc[x_test_df.index][dataClean],
    #                 'prediction': y_pred})
    # # test table
    # test_label = LabelFrame(figureFrame, text='Test Data',font=my_font1,borderwidth=3)
    # test_label.pack(side=RIGHT)
    # test_table = ttk.Treeview(test_label)
    # test_table['columns']=('tweet','prediction')
    # test_table['show']='headings'
    # for column in test_table['columns']:
    #     test_table.heading(column, text=column)
    #     test_table.column(column, width=100,stretch=False)
    # test_list = test_df.to_numpy().tolist()
    # for row in test_list:
    #     test_table.insert("","end",values=row)
    # # scrollbar
    # hs=Scrollbar(test_label,orient=HORIZONTAL,command=test_table.xview)
    # test_table.configure(xscrollcommand=hs.set)
    # hs.pack(side=BOTTOM,fill='x')
    # vs=Scrollbar(test_label,orient=VERTICAL,command=test_table.yview)
    # test_table.configure(yscrollcommand=vs.set)
    # vs.pack(side=RIGHT,fill='y')
    # test_table.pack()
    
    # report = metrics.classification_report(y_test, y_pred, target_names=['negatif', 'netral', 'positif'])
    # confm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confm,  display_labels=columns)
    # df_cm = DataFrame(confm, index=columns, columns=columns)
    # plt.switch_backend('agg')
    # fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(3, 3))
    # ax = sn.heatmap(df_cm, cmap='Greens', annot=True)
    # ax.set_title('Confusion matrix')
    # ax.set_xlabel('Label prediksi')
    # ax.set_ylabel('Label sebenarnya')
    # # Tampilkan figure di dalam canvas tkinter
    # canvas = FigureCanvasTkAgg(fig, master=figureFrame)
    # canvas.get_tk_widget().pack()
    
    # label input text
    label_input_knn=Label(inputFrame,text='input text:',font=my_font1)
    label_input_knn.grid(row=1, column=0 ,pady=4,sticky='w')
#     # input inputnbc
    entry_knn = Entry(inputFrame,font=my_font1,textvariable=inputknn,width=50 )
    entry_knn.grid(row=1, column=1 ,padx=10,sticky='w')
    knn_input_button=Button(inputFrame,text='input',command=lambda: knn_test(entry_knn.get(), bow_transformer, knn))
    knn_input_button.grid(row=2,column=0)
#     # label input text
    label_result_knn=Label(inputFrame,text='hasilnya:',font=my_font1)
    label_result_knn.grid(row=3, column=0 ,pady=4,sticky='w')
#     # input knn
    entry_knn_result = Entry(inputFrame,font=my_font1,textvariable=resultnbc,width=50 )
    entry_knn_result.grid(row=3, column=1 ,padx=10,sticky='w')
    entry_knn_result.config(state= "disabled")
    knn_input_button=Button(bFrame,text='input',command=lambda: knn_test(entry_knn.get(), bow_transformer, knn))
    knn_input_button.grid(row=2,column=0)
    cm_button=Button(bFrame,text='Confusion matrix',command=lambda: showConfusionMatrixCanvas(y_test,y_pred))
    cm_button.grid(row=1,column=0)
    cr_button=Button(bFrame,text='Classification report',command=lambda: classification_report(y_test,y_pred))
    cr_button.grid(row=1,column=1)
def knn_test(inputknn,bow_transformer, knn):
    if(inputknn==""):
        return None
    test_1_unseen =  bow_transformer.transform([inputknn])
    data=test_1_unseen.toarray()
    prediction_unseen = knn.predict(data)
    return resultnbc.set(prediction_unseen)
def is_numeric(char):
    """Validasi apakah input adalah numerik"""
    return char.isdigit()
def crawlPopUp():
    global top
    top= Toplevel(root)
    top.title("Crawling Data")
    top.geometry("500x150")
    my_font1=('times', 10, 'normal')


    # label Search
    label_search=Label(top,text='Search :',font=my_font1)
    label_search.grid(row=0, column=0,sticky='w')
    # entry search
    entry_search = Entry(top,font=my_font1,textvariable=search,width=50)
    entry_search.grid(row=0, column=1,pady=4,padx=10,sticky='w')

    # label Limit
    label_limit=Label(top,text='Limit :',font=my_font1)
    label_limit.grid(row=1, column=0,sticky='w')
    # entry Limit
    entry_limit = Entry(top,font=my_font1,textvariable=limit,width=50)
    entry_limit.grid(row=1, column=1,pady=4,padx=10,sticky='w')
    entry_limit['validate'] = 'key'
    entry_limit['validatecommand'] = (entry_limit.register(is_numeric), '%P')
    # label fileType
    label_file_type=Label(top,text='Filetype :',font=my_font1)
    label_file_type.grid(row=2, column=0,sticky='w')
    # entry Limit
    list_type=['excel','csv']
    fileTypes = ttk.Combobox(top, width = 27, values= list_type,textvariable=fileType)
    fileTypes.grid(row=2,column=1,pady=4,padx=10,sticky='w')
    fileTypes.configure(state='readonly')

    crawling_data_button=Button(top,text='Start',command=lambda:threading.Thread(
            target=CrawlingData, args=(entry_search.get(),entry_limit.get(),fileTypes.get())).start())
    crawling_data_button.grid(row=3,column=0)
def CrawlingData(search,limit,fileType):
    print(limit,search,fileType)
    
    load_dotenv()
    consumer_key = os.getenv('consumer_key')
    consumer_secret = os.getenv('consumer_secret')
    access_token = os.getenv('access_token')
    access_token_secret = os.getenv('access_token_secret')

    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        tweets = tweepy.Cursor(
                    api.search_tweets, q=search, lang="id", 
                    tweet_mode='extended').items(int(limit))
        data = {"username": [], "fulltext": [], "created_at": []}
        for tweet in tweets:
            data["username"].append(tweet.user.screen_name)
            data["fulltext"].append(tweet.full_text)
            data["created_at"].append(tweet.created_at)
        
        db_tweets=pd.DataFrame(data)
        
        print('Proses Scrapping Selesai Dengan Jumlah Data', len(db_tweets))
        files = [
                ("Excel file","*.xlsx"),
                ("CSV file","*.csv")]
        file = filedialog.asksaveasfile(mode='w',filetypes = files, defaultextension = files)
        if(file is not None):
            db_tweets['created_at'] = db_tweets['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            excel_filename = r"{}".format(file.name)
            if excel_filename[-4:] == ".csv":
                db_tweets.to_csv(file.name, index=False, encoding='utf-8')
            else:
                db_tweets.to_excel(file.name, index=False, encoding='utf-8')
        top.destroy()
    except tweepy.errors.Unauthorized as e:
        print(e)
        top.destroy()







root =Tk()
filePath= StringVar()
fileName=StringVar()
file_column=StringVar()
data_tweet=StringVar()
data_klasifikasi=StringVar()
data_clean=StringVar()
akurasi=StringVar()
akurasiKNN=StringVar()
kFeatures=StringVar()
status=StringVar()
search=StringVar()
fileType=StringVar()
limit=StringVar()
list_column=[]
inputnbc=StringVar()
resultnbc=StringVar()
inputknn=StringVar()
resultknn=StringVar() 
root.title('Analisis Sentimen dengan NBC dan KNN')
root.geometry('600x650')
root.resizable(0, 0)

# style


# font
my_font1=('times', 10, 'normal')
my_fontMainLabel=('arial',11,'bold')

# frame
innerFrame = LabelFrame(root,text='Data info',font=my_fontMainLabel)
tableFrame = Frame(root, borderwidth=3)
buttonFrame= LabelFrame(root, text='Process',font=my_fontMainLabel,relief=FLAT)
# tableFrame.pack(fill="both", expand=True)

for frame in [ innerFrame,tableFrame,buttonFrame]:
    frame.pack(padx=10,pady=10)
    # frame.pack(fill="both", expand=True)

buttonFrame.pack(side=LEFT,padx=30,pady=10)
# buttonFrame.grid(row=0,column=0,sticky=W,padx=10,pady=10)

# Menu Bar
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=browseFiles)
filemenu.add_command(label="Close", command=closeapp)
menubar.add_cascade(label="File", menu=filemenu)

# label file
label_f_path=Label(innerFrame,text='File path:',font=my_font1)
label_f_path.grid(row=0, column=0,sticky='w')

# label filename
label_f_name=Label(innerFrame,text='File name:',font=my_font1)
label_f_name.grid(row=1, column=0 ,pady=4,sticky='w')

# label c_table
label_c_table=Label(innerFrame,text='Data Table:',font=my_font1)
label_c_table.grid(row=2, column=0 ,pady=4,sticky='w')

# label data tweet
label_c_tweet=Label(innerFrame,text='Nama column data tweet:',font=my_font1)
label_c_tweet.grid(row=3, column=0 ,pady=4,sticky='w')

# label data klasifikasi
label_c_tweet=Label(innerFrame,text='Nama column data klasifikasi manual:',font=my_font1)
label_c_tweet.grid(row=4, column=0 ,pady=4,sticky='w')

# label data celan
label_c_tweet=Label(innerFrame,text='Nama column data clean:',font=my_font1)
label_c_tweet.grid(row=5, column=0 ,pady=4,sticky='w')

# label data k
label_k_features=Label(innerFrame,text='K:',font=my_font1)
label_k_features.grid(row=6, column=0 ,pady=4,sticky='w')




# input file path
entry_file_path = Entry(innerFrame,font=my_font1,textvariable=filePath,width=50 )
entry_file_path.grid(row=0, column=1 ,padx=10,sticky='w')
entry_file_path.config(state= "disabled")
# input file name
entry_f_name = Entry(innerFrame,font=my_font1,textvariable=fileName,width=50)
entry_f_name.grid(row=1, column=1,pady=4,padx=10,sticky='w')
entry_f_name.config(state= "disabled")
# input data table
entry_f_name = Entry(innerFrame,font=my_font1,textvariable=file_column,width=50)
entry_f_name.grid(row=2, column=1,pady=4)
entry_f_name.config(state= "disabled")
# input data tweet
entry_c_tweet =  ttk.Combobox(innerFrame,font=my_font1,textvariable=data_tweet,width=47,values=list_column)
entry_c_tweet.grid(row=3, column=1,pady=4)
entry_c_tweet.configure(state='readonly')

# input data klasifikasi
entry_c_klasifikasi = ttk.Combobox(innerFrame,font=my_font1,textvariable=data_klasifikasi,width=47,values=list_column)
entry_c_klasifikasi.grid(row=4, column=1,pady=4)
entry_c_klasifikasi.configure(state='readonly')

# input data clean
entry_c_cleanData = ttk.Combobox(innerFrame,font=my_font1,textvariable=data_clean,width=47,values=list_column)
entry_c_cleanData.grid(row=5, column=1,padx=10,pady=4)
entry_c_cleanData.configure(state='readonly')

# Input kFeatures
entry_k_features= Entry(innerFrame,font=my_font1,textvariable=kFeatures,width=50)
entry_k_features.grid(row=6, column=1 ,padx=10,pady=4)
entry_k_features['validate'] = 'key'
entry_k_features['validatecommand'] = (entry_k_features.register(is_numeric), '%P')

# Create TreeView
tree = ttk.Treeview(tableFrame,  selectmode="extended")
# tree.pack(padx=20)
# Process Frame

# label data klasifikasi nbc
akurasiNB=Label(buttonFrame,text='Akurasi Naive Bayes:',font=my_font1)
akurasiNB.grid(row=2, column=0 ,pady=4,sticky='w')

# label data akurasi knn
akurasi_knn=Label(buttonFrame,text='Akurasi KNN:',font=my_font1)
akurasi_knn.grid(row=3, column=0 ,pady=4,sticky='w')

# input akurasi data  Naive Bayes
entry_akurasi = Entry(buttonFrame,font=my_font1,textvariable=akurasi)
entry_akurasi.config(state= "disabled")
entry_akurasi.grid(row=2, column=1,sticky='w',pady=4)

# input akurasi data  KNN
entry_akurasi = Entry(buttonFrame,font=my_font1,textvariable=akurasiKNN)
entry_akurasi.config(state= "disabled")
entry_akurasi.grid(row=3, column=1,sticky='w',pady=4)

button_crawl_popup=Button(buttonFrame,text='Crawling Data',bd=1,command=crawlPopUp)
button_crawl_popup.grid(row=1,column=0)
# Button Preprocessing
button=Button(buttonFrame,text='preprocessing',bd=1,command=lambda:threading.Thread(
        target=preprocessing, args=( entry_file_path.get(),entry_c_tweet.get(),entry_c_klasifikasi.get())).start())
button.grid(row=1,column=1)
# Button Naive Bayes
buttonNB=Button(buttonFrame,text='klasifikasi NBC',command=lambda:threading.Thread(
        target=naiveBayes, args=( entry_file_path.get(),entry_c_tweet.get(),entry_c_klasifikasi.get(),entry_c_cleanData.get())).start())
buttonNB.grid(row=1,column=2)
# Button KNN
buttonKNN=Button(buttonFrame,text='klasifikasi KNN',command=lambda:threading.Thread(
        target=knn, args=( entry_file_path.get(),entry_c_tweet.get(),entry_c_klasifikasi.get(),entry_c_cleanData.get(),int(entry_k_features.get()))).start())
buttonKNN.grid(row=1,column=3)

# Status
statusLabel=Label(root,textvariable=status,bd=1,relief=FLAT,anchor=E)
statusLabel.pack(side=BOTTOM)

# root
root.config(menu=menubar)
root.mainloop()