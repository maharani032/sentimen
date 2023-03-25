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
# kalau dibutuhkan tamble
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
        tree.column(column, width=100,stretch=False)
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
    'yg','dgn','utk','gw','gue','tapi','kl','deh','gua','lu','lo','kalo','trs','jd','nih','ntar','nya','lg'
    ,'gk','dpt','dr','kpn','kok','kyk','dong','donk','yah','u','tuh','kpd','ksl','kzl','byar','si','siii','wkwk','wkwkwk','ini','mmg','jd',
    'wow','wowwwwwah','icymi','ni','coy','coii','isenkisenk','dg','ke','ga','pdhl','aja','tadi','krn','tak',
    'aja','sbb','is','too','kuy','se','skrg','yep','aja','as','yaa','jadinya','aja','coba','tibatiba','shit','knp','jdi','udah'
    ,'sih','bang','oke','nah','bgt','km','ttg','dlm','aaa','kang','hehe','wes','you','doang','kamu','wkkw','ong','sm','he','yeee'
    ,'emg','kak','gan','woy','dm','hi','kakk','min','di'
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
    if(len(dataTweet)<2):
        return messagebox.showerror("Information", "data tweet kosong")
    if(len(dataKlasifikasi)<2):
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
    if(len(dataTweet)<2 or len(dataClean)<2 or len(dataKlasifikasi)<2 or len(filePath)<2):
        return messagebox.showerror("Information", "data tweet kosong")
    excel_filename = r"{}".format(filePath)
    if excel_filename[-4:] == ".csv":
        df = pd.read_csv(excel_filename)
    else:
        df = pd.read_excel(excel_filename)
    status.set('Running NBC please dont close')
    statusLabel.update()
    try:
        # tdf-id
        bow_transformer = CountVectorizer().fit(df[dataClean])
        tokens = bow_transformer.get_feature_names_out()
        text_bow = bow_transformer.transform(df[dataClean])
        tfidf_transformer=TfidfTransformer().fit(text_bow)
        tweet_tfidf=tfidf_transformer.transform(text_bow)
        dd=pd.DataFrame(data=tweet_tfidf.toarray(),columns=tokens)
        # train Naive Bayes
        X = text_bow.toarray()
        x_train, x_test, y_train, y_test = train_test_split(X, df[dataKlasifikasi],test_size=0.1, random_state=35)

        # x_train, x_test, y_train, y_test = train_test_split(X, df.label,test_size=0.2, random_state=35)
        model = MultinomialNB().fit(x_train,y_train)
        prediction = model.predict(x_test)
        predict= pd.Series(prediction)
        
        # true_label= pd.Series(y_test)
        # uji naive bayes
        t = time()
        y_pred = model.predict(x_test)
        test_time = time() - t
        print("test time:  %0.3fs" % test_time)

        score1 = metrics.accuracy_score(y_test, y_pred)
        print("accuracy:   %0.3f" % score1)
        akurasi.set(score1)

        print(metrics.classification_report(y_test, y_pred, target_names=['negatif', 'netral', 'positif']))
        columns = ['negatif','netral','positif']
        confm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=confm,  display_labels=columns)
        df_cm = DataFrame(confm, index=columns, columns=columns)
        plt.switch_backend('agg')
        ax = sn.heatmap(df_cm, cmap='Greens', annot=True)

        ax.set_title('Confusion matrix')
        ax.set_xlabel('Label prediksi')
        ax.set_ylabel('Label sebenarnya')


        files = [
            ("image file","*.png")]
        file = filedialog.asksaveasfile(mode='w',filetypes = files, defaultextension = files)
        if file:
            excel_filename = r"{}".format(file.name)
            plt.savefig(file.name)
            plt.close()
        # test data
        user_data = ["salah banget"]
        test_1_unseen =  bow_transformer.transform(user_data)
        data=test_1_unseen.toarray()
        prediction_unseen = model.predict(data)
        print(prediction_unseen)



        # top1=Toplevel(width=200)
        # top1.title('Confusion Matrix')
        # plt.show()
        # my_img=ImageTk.PhotoImage()
        
        # plt.savefig('out.png')

        # status
        status.set('Ready...')
        statusLabel.update()
        return None
    except ValueError:
        return messagebox.showerror("Information",ValueError)
def knn(filePath,dataTweet,dataKlasifikasi,dataClean,k):
    if(len(dataTweet)<2 or len(dataClean)<2 or len(dataKlasifikasi)<2 or len(filePath)<2 or k is None):
        return messagebox.showerror("Information", "data tweet kosong")
    print(k)
    excel_filename = r"{}".format(filePath)
    if excel_filename[-4:] == ".csv":
        df = pd.read_csv(excel_filename)
    else:
        df = pd.read_excel(excel_filename)
    status.set('Running KNN please dont close...')
    statusLabel.update()
    # running = Label(innerFrame,text="Running knn please do not close..",font = (16))
    # running.grid(row=5, column=0 ,pady=4)
    # running.pack()
    # root.update()
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

    # # menghitung performa model pada data testing
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    akurasiKNN.set(acc)
    print(metrics.classification_report(y_test, y_pred, target_names=['negatif', 'netral', 'positif']))
def is_numeric(char):
    """Validasi apakah input adalah numerik"""
    return char.isdigit()
def crawlPopUp():
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
        
        # db_tweets = pd.DataFrame(columns=['username', 'tweetcreatedts', 'text'])
        tweets = tweepy.Cursor(
                    api.search_tweets, q=search, lang="id", 
                    tweet_mode='extended').items(int(limit))
        # tweet_list=[]
        # tweet_list = [tweet for tweet in tweets]
        data = {"username": [], "fulltext": [], "created_at": []}
        for tweet in tweets:
            data["username"].append(tweet.user.screen_name)
            data["fulltext"].append(tweet.full_text)
            data["created_at"].append(tweet.created_at)
        
        db_tweets=pd.DataFrame(data)
        #     tweetcreatedts = tweet.created_at
        #     try:
        #         text = tweet.retweeted_status.full_text
        #     except AttributeError:
        #         text = tweet.full_text
        #     ith_tweet = [username, tweetcreatedts, text]
        #     db_tweets.loc[len(db_tweets)] = ith_tweet
        
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
    except tweepy.errors.Unauthorized as e:
        print(e)
    # except tweepy.TweepError as e:
    #     print(e)






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
    # fileTypes = ttk.Combobox(top, width = 27, values= list_type,textvariable=fileType)
    # fileTypes.grid(row=2,column=1,pady=4,padx=10,sticky='w')
    # fileTypes.configure(state='readonly')
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