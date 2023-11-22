from tkinter import filedialog, messagebox, ttk
import customtkinter
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neighbors import KNeighborsClassifier




class KNNPopUp(customtkinter.CTkToplevel):
    def __init__(self,master,filepath_var=None,tweet_var=None,cleantweet_var=None,label_var=None, k_var=None,paramknn_var=None,kfold_var=None):
        super().__init__(master)
        self.filepath=filepath_var
        self.tweet=tweet_var
        self.ctweet=cleantweet_var
        self.label=label_var
        self.k=k_var
        self.kfold=kfold_var
        self.paramknn=paramknn_var

        self.kolomakurasi=customtkinter.StringVar()
        self.presisimean=customtkinter.StringVar()
        self.recallmean=customtkinter.StringVar()
        self.akurasimean=customtkinter.StringVar()
        self.akurasiknn=customtkinter.StringVar()
        self.presisiknn=customtkinter.StringVar()
        self.recallknn=customtkinter.StringVar()

        k=self.k.get()
        tweet=self.tweet.get()
        clean=self.ctweet.get()
        label=self.label.get()
        kfold=self.kfold.get()
        params=self.paramknn.get()
        print(params)
        excel_filename = r"{}".format(self.filepath.get())
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)
        if (df[label].count()!= df[tweet].count()):
            messagebox.showerror("Information", "panjang data tidak sesuai")
            return self.destroy()

        df = df.dropna(subset=[label]).reset_index(drop=True)
        # hitung term frequency (tf) dari data teks
        
        X_train, X_test, y_train, y_test = train_test_split(df[clean],df[label] , test_size=0.2
                                                            , random_state=32)
        df_train= pd.DataFrame()
        df_train['text'] = X_train
        df_train['label'] = y_train
        df_train['jenis']='latihan'

        df_test = pd.DataFrame()
        df_test['text'] = X_test
        df_test['label'] = y_test
        df_test['jenis']='test'

        # Langkah 3: Ekstraksi Fitur
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_vec )
        X_test_tfidf = tfidf_transformer.transform(X_test_vec)
        if params=='Euclidean Distance':
            modelKNN = KNeighborsClassifier(n_neighbors=int(k), metric='euclidean')
        elif params=='Manhattan Distance':
            modelKNN = KNeighborsClassifier(n_neighbors=int(k),metric='manhattan')
        elif params=='Minkowski Distance':
            modelKNN = KNeighborsClassifier(n_neighbors=int(k),metric='minkowski')
        knn=modelKNN.fit(X_train_tfidf,y_train)
        kf = KFold(n_splits=int(kfold))
        y_pred=knn.predict(X_test_tfidf)
        
        scores = cross_val_score(knn, X_train_tfidf, y_train, cv=kf,scoring='accuracy')
        precision_scores = cross_val_score(knn, X_train_tfidf, y_train, cv=kf, scoring='precision_weighted')
        recall_scores = cross_val_score(knn, X_train_tfidf, y_train, cv=kf, scoring='recall_weighted')
        
        predicted_labels = cross_val_predict(knn, X_train_tfidf, y_train, cv=kf)
        
        y_pred=knn.predict(X_test_tfidf)
        df_train['prediksi']=predicted_labels
        df_test['prediksi']=y_pred
        df_combined = pd.concat([df_train, df_test], ignore_index=False)
        

        self.kolomakurasi.set(str(scores))
        mean_score = np.array(scores).mean()  # Convert to numpy array and calculate mean

        self.akurasimean.set(str(mean_score))
        self.recallmean.set(str(recall_scores.mean()))
        self.presisimean.set(str(precision_scores.mean()))
        # Layar
        
        self.akurasiknn.set(str(metrics.accuracy_score(y_test,y_pred)))
        self.presisiknn.set(str(metrics.precision_score(y_test,y_pred, average='weighted')))
        self.recallknn.set(str(metrics.recall_score(y_test,y_pred, average='weighted')))

        self.grid_columnconfigure(1, weight = 1)
        self.grid_rowconfigure(0, weight = 1)
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

        self.label_akurasi = customtkinter.CTkLabel(self.dataframe, text='Akurasi dengan K-Fold Cross Validation:')
        self.label_akurasi.grid(row=1,column=0,padx=10,pady=4,sticky='w')
        self.emptyakurasi=customtkinter.CTkEntry(self.dataframe,textvariable=self.kolomakurasi,width=200)
        self.emptyakurasi.grid(row=1,column=0,pady=4,padx=4,sticky='w')
        self.emptyakurasi.configure(state= "disabled")

        self.label_meanaccuracy=customtkinter.CTkLabel(self.dataframe, width=100,text='Rata-rata akurasi k-fold:')
        self.label_meanaccuracy.grid(row=2,column=0,padx=10,pady=4,sticky='w')
        self.emptymean=customtkinter.CTkEntry(self.dataframe,textvariable=self.akurasimean,width=200)
        self.emptymean.grid(row=2,column=1,pady=4,padx=4,sticky='w')
        self.emptymean.configure(state= "disabled")

        self.label_meanrecall=customtkinter.CTkLabel(self.dataframe, width=100,text='Rata-rata recall k-fold:')
        self.label_meanrecall.grid(row=3,column=0,padx=10,pady=4,sticky='w')
        self.emptyrecall_mean=customtkinter.CTkEntry(self.dataframe,textvariable=self.recallmean,width=200)
        self.emptyrecall_mean.grid(row=3,column=1,pady=4,padx=4,sticky='w')
        self.emptyrecall_mean.configure(state= "disabled")
        
        self.label_mean_presisi=customtkinter.CTkLabel(self.dataframe, width=100,text='Rata-rata presisi k-fold:')
        self.label_mean_presisi.grid(row=4,column=0,padx=10,pady=4,sticky='w')
        self.mean_presisi=customtkinter.CTkEntry(self.dataframe,textvariable=self.presisimean,width=200)
        self.mean_presisi.grid(row=4,column=1,pady=4,padx=4,sticky='w')
        self.mean_presisi.configure(state= "disabled")

        self.label_wo=customtkinter.CTkLabel(self.dataframe,text="Klasifikasi dengan K-Nearest Neighbors",font=customtkinter.CTkFont(weight='bold'))
        self.label_wo.grid(row=5,column=0,padx=10,sticky='w')

        self.label_akurasinbc = customtkinter.CTkLabel(self.dataframe, text='Akurasi dengan K-Nearest Neighbors:')
        self.label_akurasinbc.grid(row=6,column=0,padx=10,pady=4,sticky='w')
        self.emptyakurasi_nbc=customtkinter.CTkEntry(self.dataframe,textvariable=self.akurasiknn,width=200)
        self.emptyakurasi_nbc.grid(row=6,column=1,pady=4,padx=4,sticky='w')
        self.emptyakurasi_nbc.configure(state= "disabled")

        self.label_nbcrecall=customtkinter.CTkLabel(self.dataframe,text='Recall dengan K-Nearest Neighbors:')
        self.label_nbcrecall.grid(row=7,column=0,padx=10,pady=4,sticky='w')
        self.emptyrecall_nbc=customtkinter.CTkEntry(self.dataframe,textvariable=self.recallknn,width=200)
        self.emptyrecall_nbc.grid(row=7,column=1,pady=4,padx=4,sticky='w')
        self.emptyrecall_nbc.configure(state= "disabled")
        
        self.label_nbc_presisi=customtkinter.CTkLabel(self.dataframe, text='Presisi dengan K-Nearest Neighbors:')
        self.label_nbc_presisi.grid(row=8,column=0,padx=10,pady=4,sticky='w')
        self.nbc_presisi=customtkinter.CTkEntry(self.dataframe,textvariable=self.presisiknn,width=200)
        self.nbc_presisi.grid(row=8,column=1,pady=4,padx=4,sticky='w')
        self.nbc_presisi.configure(state= "disabled")

        confm = metrics.confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(4, 4))
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
        # classificationTop.title('Classification Report')
        self.textClassification=customtkinter.CTkLabel(self.confusionmatrix,text='Classification Report',font=customtkinter.CTkFont(weight='bold'))
        self.textClassification.grid(row=0,column=0,padx=10,sticky='nsew')
        text = customtkinter.CTkTextbox(self.confusionmatrix,corner_radius=5,width=300)
        text.grid(row=1,column=0,padx=10,pady=4,sticky='nsew') 
        # text.pack()
        report = metrics.classification_report(y_test, y_pred)
        text.insert("0.0", report)



