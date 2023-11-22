from tkinter import filedialog, messagebox, ttk
import customtkinter
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_predict




class NaivePopUp(customtkinter.CTkToplevel):
    def __init__(self,master,filepath_var=None,tweet_var=None,cleantweet_var=None
                 ,label_var=None,nb_params=None,kfold_var=None):
        super().__init__(master)
        self.filepath=filepath_var
        self.tweet=tweet_var
        self.ctweet=cleantweet_var
        self.label=label_var
        self.params=nb_params
        self.kfolds=kfold_var

        self.akurasi=customtkinter.StringVar()
        self.presisimean=customtkinter.StringVar()
        self.recallmean=customtkinter.StringVar()
        self.akurasimean=customtkinter.StringVar()
        self.akurasinbc=customtkinter.StringVar()
        self.presisinbc=customtkinter.StringVar()
        self.recallnbc=customtkinter.StringVar()

        tweet=self.tweet.get()
        clean=self.ctweet.get()
        label=self.label.get()
        params=self.params.get()
        kfold=self.kfolds.get()
        print(params)
        excel_filename = r"{}".format(self.filepath.get())
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)
        if (df[label].count()!= df[tweet].count()):
            print('tidak sama')
            messagebox.showerror("Information", "panjang data tidak sesuai")
            return self.destroy()

        df = df.dropna(subset=[label]).reset_index(drop=True)
        
        X_train, X_test, y_train, y_test = train_test_split(df[clean],df[label], test_size=0.2,random_state=32)
        df_train= pd.DataFrame()
        df_train['text'] = X_train
        df_train['label'] = y_train
        df_train['jenis']='latihan'

        df_test = pd.DataFrame()
        df_test['text'] = X_test
        df_test['label'] = y_test
        df_test['jenis']='test'


        vectorizer = CountVectorizer()
        train_X_tfidf = vectorizer.fit_transform(df_train['text'])
        test_X_tfidf = vectorizer.transform(df_test['text'])
        tfidf_transformer = TfidfTransformer()
        train_X_tfidf = tfidf_transformer.fit_transform(train_X_tfidf )
        test_X_tfidf = tfidf_transformer.transform(test_X_tfidf)
        if params=='ComplementNB':
            model_NB = ComplementNB() #inisialiasi library
        elif params=='MultinomialNB':
            model_NB=MultinomialNB()

        NB2 = model_NB.fit(train_X_tfidf,y_train) #melatih model menggunakan data pelatihan
        kf = KFold(n_splits=int(kfold))
        scores=cross_val_score(NB2, train_X_tfidf, y_train, cv=kf, scoring='accuracy')
        precision_scores = cross_val_score(NB2, train_X_tfidf, y_train, cv=kf, scoring='precision_weighted')
        recall_scores = cross_val_score(NB2, train_X_tfidf, y_train, cv=kf, scoring='recall_weighted')
        predicted_labels = cross_val_predict(NB2, train_X_tfidf, y_train, cv=kf)
        
        y_pred=NB2.predict(test_X_tfidf)
        df_train['prediksi']=predicted_labels
        df_test['prediksi']=y_pred
        df_combined = pd.concat([df_train, df_test], ignore_index=False)

        self.akurasi.set(str(scores))
        mean_score = np.array(scores).mean()  # Convert to numpy array and calculate mean

        self.akurasimean.set(str(mean_score))
        self.recallmean.set(str(recall_scores.mean()))
        self.presisimean.set(str(precision_scores.mean()))

        self.akurasinbc.set(str(metrics.accuracy_score(y_test,y_pred)))
        self.presisinbc.set(str(metrics.precision_score(y_test,y_pred, average='weighted')))
        self.recallnbc.set(str(metrics.recall_score(y_test,y_pred, average='weighted')))
        
        self.grid_rowconfigure(0, weight = 1)
        self.grid_rowconfigure(1, weight = 1)

        self.dataframe=customtkinter.CTkFrame(self)
        self.dataframe.grid(row=0,column=0,padx=10,pady=10,sticky='n')
        self.grafis=customtkinter.CTkFrame(self)
        self.grafis.grid(row=0,column=1,padx=10,pady=10,sticky='n')
        self.tabel=customtkinter.CTkFrame(self,width=100)
        self.tabel.grid(row=1,column=0,padx=10,pady=10,sticky='nsew')
        self.confusionmatrix=customtkinter.CTkFrame(self)
        self.confusionmatrix.grid(row=1,column=1,pady=10,padx=10,sticky='nsew')


        self.label_data=customtkinter.CTkLabel(self.dataframe,text="K-fold Cross Validation",font=customtkinter.CTkFont(weight='bold'))
        self.label_data.grid(row=0,column=0,padx=10,sticky='w')

        self.label_akurasi = customtkinter.CTkLabel(self.dataframe, text='Akurasi data latih dengan K-Fold Cross Validation:')
        self.label_akurasi.grid(row=1,column=0,padx=10,pady=4,sticky='w')
        self.emptyakurasi=customtkinter.CTkEntry(self.dataframe,textvariable=self.akurasi,width=200)
        self.emptyakurasi.grid(row=1,column=1,pady=4,padx=4,sticky='w')
        self.emptyakurasi.configure(state= "disabled")

        self.label_meanaccuracy=customtkinter.CTkLabel(self.dataframe, width=100,text='Rata-rata akurasi data latih k-fold:')
        self.label_meanaccuracy.grid(row=2,column=0,padx=10,pady=4,sticky='w')
        self.emptymean=customtkinter.CTkEntry(self.dataframe,textvariable=self.akurasimean,width=200)
        self.emptymean.grid(row=2,column=1,pady=4,padx=4,sticky='w')
        self.emptymean.configure(state= "disabled")

        self.label_meanrecall=customtkinter.CTkLabel(self.dataframe, width=100,text='Rata-rata data latih recall k-fold:')
        self.label_meanrecall.grid(row=3,column=0,padx=10,pady=4,sticky='w')
        self.emptyrecall_mean=customtkinter.CTkEntry(self.dataframe,textvariable=self.recallmean,width=200)
        self.emptyrecall_mean.grid(row=3,column=1,pady=4,padx=4,sticky='w')
        self.emptyrecall_mean.configure(state= "disabled")
        
        self.label_mean_presisi=customtkinter.CTkLabel(self.dataframe, width=100,text='Rata-rata presisi data latih dengan k-fold:')
        self.label_mean_presisi.grid(row=4,column=0,padx=10,pady=4,sticky='w')
        self.mean_presisi=customtkinter.CTkEntry(self.dataframe,textvariable=self.presisimean,width=200)
        self.mean_presisi.grid(row=4,column=1,pady=4,padx=4,sticky='w')
        self.mean_presisi.configure(state= "disabled")

        self.label_wo=customtkinter.CTkLabel(self.dataframe,text="Klasifikasi Naive Bayes",font=customtkinter.CTkFont(weight='bold'))
        self.label_wo.grid(row=5,column=0,padx=10,sticky='w')

        self.label_akurasinbc = customtkinter.CTkLabel(self.dataframe, text='Akurasi dengan Naive Bayes:')
        self.label_akurasinbc.grid(row=6,column=0,padx=10,pady=4,sticky='w')
        self.emptyakurasi_nbc=customtkinter.CTkEntry(self.dataframe,textvariable=self.akurasinbc,width=200)
        self.emptyakurasi_nbc.grid(row=6,column=1,pady=4,padx=4,sticky='w')
        self.emptyakurasi_nbc.configure(state= "disabled")

        self.label_nbcrecall=customtkinter.CTkLabel(self.dataframe,text='Recall dengan Naive Bayes:')
        self.label_nbcrecall.grid(row=7,column=0,padx=10,pady=4,sticky='w')
        self.emptyrecall_nbc=customtkinter.CTkEntry(self.dataframe,textvariable=self.recallnbc,width=200)
        self.emptyrecall_nbc.grid(row=7,column=1,pady=4,padx=4,sticky='w')
        self.emptyrecall_nbc.configure(state= "disabled")
        
        self.label_nbc_presisi=customtkinter.CTkLabel(self.dataframe, text='Presisi dengan Naive Bayes:')
        self.label_nbc_presisi.grid(row=8,column=0,padx=10,pady=4,sticky='w')
        self.nbc_presisi=customtkinter.CTkEntry(self.dataframe,textvariable=self.presisinbc,width=200)
        self.nbc_presisi.grid(row=8,column=1,pady=4,padx=4,sticky='w')
        self.nbc_presisi.configure(state= "disabled")

        # if jenislabel.isn
        confm = metrics.confusion_matrix(y_test, y_pred)
        # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confm)
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(4, 4))
        # plt.xlabel('Prediksi', fontsize=18)
        ax = sn.heatmap(confm, cmap='Greens', annot=True, fmt='d')
        ax.set_title('Confusion matrix')
        ax.set_xlabel('Label prediksi')
        ax.set_ylabel('Label sebenarnya')

        canvas=FigureCanvasTkAgg(fig,self.grafis)
        canvas.get_tk_widget().grid(row=0,column=0,padx=10)

        self.tree=ttk.Treeview(self.tabel,selectmode='extended')
        self.tree["column"] = list(df_combined.columns)
        # self.tree["show"] = "headings"
        for column in self.tree["columns"]:
            self.tree.heading(column, text=column,anchor='w')
            self.tree.column(column,anchor='w',width=100,stretch=False)

        df_rows = df_combined.to_numpy().tolist() # turns the dataframe into a list of lists
        for row in df_rows:
            self.tree.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
        hs=ttk.Scrollbar(self.tabel, orient="horizontal", command=self.tree.xview)
        hs.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=hs.set)
        ttk.Style().configure("TreeviewItem", rowheight = 50, font = (None, 50))
        vs=ttk.Scrollbar(self.tabel, orient="vertical", command=self.tree.yview)
        vs.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.grid(row=0, column=0, sticky="nsew",padx=0,pady=0)

        self.textClassification=customtkinter.CTkLabel(self.confusionmatrix,text='Classification Report',font=customtkinter.CTkFont(weight='bold'))
        self.textClassification.grid(row=0,column=0,padx=10,sticky='nsew')
        text = customtkinter.CTkTextbox(self.confusionmatrix,corner_radius=5,width=300)
        text.grid(row=1,column=0,padx=10,pady=4,sticky='nsew') 
        # text.pack()
        report = metrics.classification_report(y_test, y_pred)
        text.insert("0.0", report)



