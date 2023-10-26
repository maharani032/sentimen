from tkinter import messagebox
import customtkinter
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neighbors import KNeighborsClassifier

class bandingPopUp(customtkinter.CTkToplevel):
    def __init__(self,master,filepath_var=None,tweet_var=None,cleantweet_var=None,label_var=None):
        super().__init__(master)
        self.filepath=filepath_var
        self.tweet=tweet_var
        self.ctweet=cleantweet_var
        self.label=label_var

        tweet=self.tweet.get()
        clean=self.ctweet.get()
        label=self.label.get()

        excel_filename = r"{}".format(self.filepath.get())
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)
        if (df[label].count()!= df[tweet].count()):
            messagebox.showerror("Information", "panjang data tidak sesuai")
            return self.destroy()

        df = df.dropna(subset=[label]).reset_index(drop=True)
        bow_transformer = CountVectorizer().fit(df[clean])
        tokens = bow_transformer.get_feature_names_out()
        text_bow = bow_transformer.transform(df[clean])
        data = pd.DataFrame(text_bow.toarray(), columns=tokens)

        tfidf_transformer=TfidfTransformer().fit(text_bow)
        tweet_tfidf=tfidf_transformer.transform(text_bow)

        jenislabel=df[label].unique()
        print(jenislabel)
        X = text_bow.toarray()
        Y=df[label]
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X,Y , test_size=0.2,random_state=32)
        X_train, X_test, y_train, y_test = train_test_split(df[clean],df[label] ,test_size=0.2
                                                            , random_state=32)
        
        # Langkah 3: Ekstraksi Fitur
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_vec )
        X_test_tfidf = tfidf_transformer.transform(X_test_vec)

        k_values = [1,2,3,4,5,6,7,8,9,10]
        knn_accuracies = []
        nb_accuracies = []
        for k in k_values:
            # Inisialisasi dan latih model KNN
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_tfidf, y_train)

            # Prediksi dan hitung akurasi
            knn_pred = knn.predict(X_test_tfidf)
            knn_accuracy = metrics.accuracy_score(y_test, knn_pred)
            knn_accuracies.append(knn_accuracy)

            alpha = 1.0  # parameter  Laplacian Smoothing, adalah teknik yang digunakan dalam algoritma Naive Bayes untuk menghindari probabilitas nol. membantu meningkatkan kinerja algoritma dan mengurangi overfitting.

            modelNB = MultinomialNB(alpha=alpha).fit(X_train_nb,y_train_nb)

            y_pred=modelNB.predict(X_test_nb)
            # Prediksi dan hitung akurasi
            # nb_pred = nb.predict(X_train_nb)
            nb_accuracy = metrics.accuracy_score(y_test_nb, y_pred)
            nb_accuracies.append(nb_accuracy)
        # plt.figure(figsize=(10, 6))
       
        # plt.show()
        # print(knn_accuracies)
        self.grafis=customtkinter.CTkFrame(self)
        self.grafis.grid(row=0,column=1,padx=10,pady=10,sticky='n')
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(10, 6))
        plt.plot(k_values, knn_accuracies, marker='o', label='K-Nearest Neighbors (KNN)')
        plt.plot(k_values, nb_accuracies, marker='o', label='Naive Bayes (NB)')
        plt.xlabel('Nilai Parameter K (KNN) atau parameter lainnya (NB)')
        plt.ylabel('Akurasi')
        plt.title('Perbandingan Akurasi antara KNN dan Naive Bayes')
        plt.legend()
        plt.grid(True)
        canvas=FigureCanvasTkAgg(fig,self.grafis)
        canvas.get_tk_widget().grid(row=0,column=0,padx=10)