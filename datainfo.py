import re
import threading
import tkinter
import customtkinter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import threading
import asyncio
import pandas as pd
from tkinter import messagebox
import os
from tkinter import filedialog
import numpy as np
import nltk
import Sastrawi
print(Sastrawi.__file__)
from banding import bandingPopUp
from naive import NaivePopUp
from knn import KNNPopUp
from crawling import CrawlingFrame


class DataInfoFrame(customtkinter.CTkFrame):
    def update_list_column(self, *args):
        # Clear the list_column
        self.list_column.clear()

        # Get the current value of self.datatweet and split it by comma
        current_value = self.datatweet.get()
        if current_value:
            self.list_column = current_value.split(",")

        # Set the values of the ComboBox
        self_entry_datatweet = self.children['!ctkcombobox']
        self_entry_datatweet.configure(values=self.list_column)

        self_entry_datalabel = self.children['!ctkcombobox2']
        self_entry_datalabel.configure(values=self.list_column)

        self_entry_databersih = self.children['!ctkcombobox3']
        self_entry_databersih.configure(values=self.list_column)
    def is_numeric(self,char):
        return char.isdigit()
    
    def __init__(self, master,header_name="Data Info",filename_var=None,k_var=None, listcolumn_var=None,filepath_var=None, datafile_var=None, **kwargs):
        super().__init__(master, **kwargs)
        self.header_name = header_name
        # frame
        self.button_frame = customtkinter.CTkFrame(self)
        self.data=customtkinter.CTkFrame(self)
        validation = self.register(self.is_numeric)

        self.header = customtkinter.CTkLabel(self, text=self.header_name,font=customtkinter.CTkFont(weight='bold'))
        self.header.grid(row=0, column=0, padx=10, pady=10)

        self.filepath = filepath_var
        self.filename = filename_var
        self.datafile= datafile_var
        self.datatweet=listcolumn_var
        # self.k=k_var

        self.tweet=customtkinter.StringVar(value='')
        self.datalabel=customtkinter.StringVar(value='')
        self.cleantweet=customtkinter.StringVar(value='')
        self.listcolumn=customtkinter.StringVar()
        self.k=customtkinter.StringVar()
        self.kfold=customtkinter.StringVar()
        self.params_nb=customtkinter.StringVar(value='MultinomialNB')
        self.params_knn=customtkinter.StringVar(value= 'Euclidean Distance')
        self.datatweet.trace_add('write', self.update_list_column)
        
        self.list_column = [""]
        self.list_naive=[
            'MultinomialNB',
            'ComplementNB'
        ]
        self.list_knn=[
            'Euclidean Distance',
            'Manhattan Distance',
            # 'Minkowski Distance'
        ]

        self.crawling_popup=None        
        self.naive_popup=None
        self.knn_popup=None
        self.banding_popup=None

        
        self.label_filepath=customtkinter.CTkLabel(self,text='File path:')
        self.label_filepath.grid(row=1, column=0 ,padx=10,sticky='w')
        self.entry_file_path = customtkinter.CTkEntry(self,textvariable=self.filepath ,width=200 )
        self.entry_file_path.grid(row=1, column=1 ,padx=10,sticky='w')
        self.entry_file_path.configure(state= "disabled")

        
        self_label_filename=customtkinter.CTkLabel(self,text='File name:')
        self_label_filename.grid(row=2, column=0 ,padx=10 ,pady=4,sticky='w')
        entry_filename = customtkinter.CTkEntry(self,textvariable=self.filename,width=200)
        entry_filename.grid(row=2, column=1,pady=4,padx=10,sticky='w')
        entry_filename.configure(state= "disabled")

        self_label_list_tabel=customtkinter.CTkLabel(self,text='Data Tabel:')
        self_label_list_tabel.grid(row=3, column=0  ,padx=10,pady=4,sticky='w')
        self_entry_datafile =customtkinter.CTkEntry(self,textvariable=self.datafile,width=200)
        self_entry_datafile.grid(row=3, column=1,pady=4)
        self_entry_datafile.configure(state= "disabled")

        self_label_tweet=customtkinter.CTkLabel(self,text='Data tweet:')
        self_label_tweet.grid(row=4, column=0  ,padx=10,pady=4,sticky='w')
        self_entry_datatweet =  customtkinter.CTkComboBox(self,width=200,variable=self.tweet,values=self.list_column)
        self_entry_datatweet.grid(row=4, column=1,pady=4)
        self_entry_datatweet.configure(state='readonly')
        # self_entry_datatweet.set("None")

        self_label_manual=customtkinter.CTkLabel(self,text='Data label:')
        self_label_manual.grid(row=5, padx=10,column=0 ,pady=4,sticky='w')
        self_entry_datalabel =  customtkinter.CTkComboBox(self,width=200,
                                                          variable=self.datalabel,
                                                          values=self.list_column)
        self_entry_datalabel.grid(row=5, column=1,pady=4)
        self_entry_datalabel.configure(state='readonly')

        self_label_clean_tweet=customtkinter.CTkLabel(self,text='Data bersih:')
        self_label_clean_tweet.grid(row=6, padx=10,column=0 ,pady=4,sticky='w')
        self_entry_databersih =  customtkinter.CTkComboBox(self,width=200,variable=self.cleantweet,values=self.list_column)
        self_entry_databersih.grid(row=6, column=1,pady=4)
        self_entry_databersih.configure(state='readonly')

        self_label_kfold=customtkinter.CTkLabel(self,text='K Fold:')
        self_label_kfold.grid(row=7, column=0 ,padx=10 ,pady=4,sticky='w')
        entry_kfold = customtkinter.CTkEntry(self,textvariable=self.kfold,width=200,validate="key", validatecommand=(validation, '%S'))
        entry_kfold.grid(row=7, column=1,pady=4,padx=10,sticky='w')

        self_label_k=customtkinter.CTkLabel(self,text='K pada KNN:')
        self_label_k.grid(row=8, column=0 ,padx=10 ,pady=4,sticky='w')
        entry_k = customtkinter.CTkEntry(self,textvariable=self.k,width=200,validate="key", validatecommand=(validation, '%S'))
        entry_k.grid(row=8, column=1,pady=4,padx=10,sticky='w')

       
        self_label_params_nb=customtkinter.CTkLabel(self,text='Parameter Naive:')
        self_label_params_nb.grid(row=9, column=0  ,padx=10,pady=4,sticky='w')
        self_entry_params_nb =  customtkinter.CTkComboBox(self,width=200,variable=self.params_nb,values=self.list_naive)
        self_entry_params_nb.grid(row=9, column=1,pady=4)
        self_entry_params_nb.configure(state='readonly')

        self_label_params_knn=customtkinter.CTkLabel(self,text='Jarak KNN:')
        self_label_params_knn.grid(row=10, column=0  ,padx=10,pady=4,sticky='w')
        self_entry_params_knn =  customtkinter.CTkComboBox(self,width=200,variable=self.params_knn,values=self.list_knn)
        self_entry_params_knn.grid(row=10, column=1,pady=4)
        self_entry_params_knn.configure(state='readonly')

        
        self.button_crawling=customtkinter.CTkButton(master=self,text="Crawling"
                                                     ,width=100,corner_radius=10,
                                                     command=self.crawlingData)
        self.button_crawling.grid(row=11,column=0,sticky='nwse',padx=20,pady=4,columnspan=2)
        self.button_preprocessing=customtkinter.CTkButton(master=self,text="Preprocessing",
                                                          width=100,corner_radius=10,
                                                        #   command=self.preprocessing)
                                                          
        command=lambda:threading.Thread(target=self.preprocessing).start())
        self.button_preprocessing.grid(row=12,column=0,sticky='nwse',padx=20,pady=2,columnspan=2)
        self.button_naive=customtkinter.CTkButton(master=self,text="Klasifikasi Naive Bayes"
                                                  ,width=100,corner_radius=10,
                                                  command=self.naivebayes)
        self.button_naive.grid(row=13,column=0,sticky='nwse',padx=20,pady=2,columnspan=2)
        self.button_knn=customtkinter.CTkButton(master=self,text="Klasifikasi K-Nearest Neighbors"
                                                ,width=100,corner_radius=10,
                                                command=self.knn)
        self.button_knn.grid(row=14,column=0,sticky='nwse',padx=20,pady=2,columnspan=2)
        self.button_perbandingan=customtkinter.CTkButton(master=self,text="Perbanding KNN & Naive Bayes"
                                                ,width=100,corner_radius=10,
                                                command=self.banding)
        # self.button_perbandingan.grid(row=12,column=0,sticky='nwse',padx=20,pady=2,columnspan=2)

    def banding(self):
        if self.datalabel.get()=="" or self.tweet.get()=="" or self.cleantweet.get()=="":
            return messagebox.showinfo("Information","Data kosong")
        if self.banding_popup is None or not self.banding_popup.winfo_exists():
            self.banding_popup = bandingPopUp(self,cleantweet_var=self.cleantweet
                                          ,tweet_var=self.tweet,label_var=self.datalabel,
                                            filepath_var=self.filepath) # create window if its None or destroyed
            self.banding_popup.title("Klasifikasi Naive Bayes") 
        else:
            self.banding_popup.focus()
    
    def crawlingData(self):
        if self.crawling_popup is None or not self.crawling_popup.winfo_exists():
            self.crawling_popup = CrawlingFrame(self) # create window if its None or destroyed
            self.crawling_popup.title("Crawling Data") 
        else:
            self.crawling_popup.focus()
    def case_folding(self,tweet):
        return tweet.lower()
    
    def remove_mention(self,tweet,pattern_regex):
        r=re.findall(pattern_regex,tweet)
        for i in r:
            tweet=re.sub(i,"",tweet)
        return tweet
    
    def removeHastag(self,tweet):
    #     hastag dolar sign
        tweet=re.sub(r'\$\w*','',tweet)
    #     retweet lama
        tweet=re.sub(r'^RT[\s]+','',tweet)
    #     hastag
        tweet=re.sub(r'#', '',tweet)
    #     hapus angka
        tweet=re.sub('[0-9]+','',tweet)
        return tweet
    
    def remove_http(self,tweet):
        tweet=" ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
        return tweet
    
    async def clean_tweets(self,tweet):
        stop_factory = StopWordRemoverFactory().get_stop_words()
        stopwords_indonesia=nltk.corpus.stopwords.words('indonesian')
        
        # stopwords_indonesia=stopwords.words('indonesian')
        more_stopword = [
        'yg','dgn','kan','huh','bruh','xd','xf','hahaha','nya','wkwkwkkwkwk','akwkaskaksawska','wkwkw','wkwkkwwkk'
        'kok','kyk','dong','donk','yah','tuh','si','siii','wkwk','wkwkwk','ini','tp','utk','sj','pd','gw','gua','gwe',
        'wow','wowwwwwah','icymi','ni','coy','coii','isenkisenk','dg','pdhl','aja','tadi','krn','tak',
        'aja','sbb','kuy','se','skrg','yep','aja','as','yaa','jadinya','aja','coba','tibatiba','knp','jdi','dmn','kyk'
        'sih','bang','oke','nah','bgt','km','ttg','dlm','aaa','kang','hehe','wes','you','doang','wkkw','ong','sm','he','yeee'
        ,'emg','kak','gan','woy','dm','hi','kakk','min','di','noh','gais','lah','xfxf','nak','bro','x','ahhh','gasss','hmmm','sat','set','yukkkkk'
        ,'smh','eh','ni','laaah',' aihihi','fafifu','akwkaskaksawska ','xixi','kah','tbtb','bg','jg','pas','w','jga','cm','hiks','mennn','sii','sy','aku'
        ,'kan','klu','yak','smh','sj','jd','sja','jdi','fafifu','trs','tff','sih','nih','xd','d','dr','ea','ha','lu','hfft','ato','lg','jg'
        ,'gue','ku','ama','mu','anjgg','j','g','xixixi','ama','blablabla','ka','lhooo','ddd','wkwkkwkw','hua','awokaowk','ih','dek','wkwkkwwkk','tu','hihu',
        'wkwkw','ygy','cok','aihihi','deh','nder','kmu','hihi','jir','wkakakakaka','aowkawkwwk','hehehe','neng','haha','huhu','in','kek','cuy','dah','wkek'
        ,'woyy','ah','wkwkwkwkwkwkwkwkwwk','blabla','wkkwkkw','dehh','wkwkwkwk','wkwkwkw','sihhh'
        ,'poll','huft','wakaka','cui','hhh','wakakakak','hehee','hmmm','wkw','loh','awoakwoak','bund','wkwkwkwkwkwwkkwk','twt','wkkwk','sksksks','ehhh','lah'
        ,'gaes','wkwkkw','bun','an','huehuehue','aing','euy','kaga','bae','weh','wahh','nyihihihi','woi','kakakka','lha','hahahah','jos'
        ,'ges','yee','lahhh','ntah','fess','lol','ckckckck','jiakhhh','twitt','bahahaaha','hahahahaha','guys','teh','cmiiw','tweet','kwkwkwkwkw'
        ,'yahhh'
        ]
        stopword=stop_factory+stopwords_indonesia+more_stopword
        dictionary=ArrayDictionary(stopword)
        str=StopWordRemover(dictionary)

        factory=StemmerFactory()
        stemmer=factory.create_stemmer()
    #     tokenize tweets
        tokenizer=TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
        tweet_token=tokenizer.tokenize(tweet)
        tokenizer=tweet_token
        tweet_clean=[]
        stopwords=[]

        for word in tweet_token:
            if(word not in stopword):
                stopwords.append(word)
                stem_word=stemmer.stem(word)
                tweet_clean.append(stem_word)
        return  tokenizer,stopwords,tweet_clean
    
    def jointweet(self,tweet):
        return " ".join(tweet)
    def preprocessing_onclose(self):
        print(self.stop_signal)
        quit = tkinter.messagebox.askokcancel(title="Peringatan", message= "Apa anda yakin?")
        if quit:
            self.button_crawling.configure(state='normal')
            self.button_preprocessing.configure(state='normal')
            self.button_naive.configure(state='normal')
            self.button_knn.configure(state='normal')
            self.preprocessing_pop_up.destroy()
            self.stop_signal.set()
            print(self.stop_signal)
              # Set the stop signal to stop the thread

    def preprocessing(self):
        self.stop_signal = threading.Event()
        print(self.stop_signal)
        self.button_crawling.configure(state='disabled')
        self.button_preprocessing.configure(state='disabled')
        self.button_naive.configure(state='disabled')
        self.button_knn.configure(state='disabled')
        label = self.get_entry_label()
        tweet = self.get_entry_tweet()
        
        print(label)
        if(label=="" or tweet==""):
            self.button_crawling.configure(state='normal')
            self.button_preprocessing.configure(state='normal')
            self.button_naive.configure(state='normal')
            self.button_knn.configure(state='normal')
            return messagebox.showinfo("Data label atau tweet","Data label atau tweet harus diisi")
        
        excel_filename = r"{}".format(self.filepath.get())
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)
        self.preprocessing_pop_up=customtkinter.CTkToplevel(self)
        self.preprocessing_pop_up.title("Do not close")
        self.textlabel=customtkinter.CTkLabel(self.preprocessing_pop_up,text="Do not close",width=300,font=customtkinter.CTkFont(weight='bold',size=15))
        self.textlabel.grid(row=0,column=0,sticky='nesw',padx=10,pady=10)
        self.preprocessing_pop_up.focus()
        try:
            if df[label].empty or df[tweet].empty or (df[label].count() != df[tweet].count()):
                self.button_crawling.configure(state='normal')
                self.button_preprocessing.configure(state='normal')
                self.button_naive.configure(state='normal')
                self.button_knn.configure(state='normal')
                raise ValueError("Panjang data tidak sesuai")
        except ValueError as e:
            messagebox.showerror("Information", str(e))

        df['case_folding']=df[tweet].apply(lambda x:self.case_folding(x))
        df['remove_mention']=np.vectorize(self.remove_mention)(df['case_folding']," *RT* | *@[\w]*")
        df['remove_http']=df['remove_mention'].apply(lambda x:self.remove_http(x))
        df['remove_hastag']=df['remove_http'].apply(lambda x:self.removeHastag(x))
        # self.preprocessing_pop_up.protocol("WM_DELETE_WINDOW", self.preprocessing_onclose())
        self.preprocessing_pop_up.protocol("WM_DELETE_WINDOW", self.preprocessing_onclose)

        df['tokenizing'],df['stopwords'], df['stemming'] = zip(*df['remove_hastag'].apply(lambda x:asyncio.run(self.clean_tweets(x))))
        # df['tokenizer']= df['case_folding'].apply(lambda x:asyncio.run(self.clean_tweets(x)))
        
        df['cleantweet']=df['stemming'].apply(lambda x:self.jointweet(x))
        
        df.drop_duplicates(subset='cleantweet',keep='first',inplace=True)
        df.dropna()
        
        self.button_crawling.configure(state='normal')
        self.button_preprocessing.configure(state='normal')
        self.button_naive.configure(state='normal')
        self.button_knn.configure(state='normal')
        self.preprocessing_pop_up.destroy()
       
        # Save File
        files = [
                ("Excel file","*.xlsx"),
                ("CSV file","*.csv")]
        file = filedialog.asksaveasfile(mode='w',filetypes = files, defaultextension = files)
        if(file is not None):
            excel_filename = r"{}".format(file.name)
            if excel_filename[-4:] == ".csv":
                df.to_csv(file.name, index=False)
            else:
                df.to_excel(file.name, index=False)

    def naivebayes(self):
        if self.datalabel.get()=="" or self.tweet.get()=="" or self.cleantweet.get()=="" or self.kfold.get()=="":
            return messagebox.showinfo("Information","Data kosong")
        if self.naive_popup is None or not self.naive_popup.winfo_exists():
            self.naive_popup = NaivePopUp(self,cleantweet_var=self.cleantweet
                                          ,tweet_var=self.tweet,label_var=self.datalabel,
                                            filepath_var=self.filepath,nb_params=self.params_nb,kfold_var=self.kfold) # create window if its None or destroyed
            self.naive_popup.title("Klasifikasi Naive Bayes") 
        else:
            self.naive_popup.focus()
    
    def knn(self):
        if self.datalabel.get()=="" or self.tweet.get()=="" or self.cleantweet.get()=="" or self.k.get()=="" or self.kfold.get()=="":
            return messagebox.showinfo("Information","Data kosong")
        if self.knn_popup is None or not self.knn_popup.winfo_exists():
            print(self.k.get())
            self.knn_popup = KNNPopUp(self,cleantweet_var=self.cleantweet
                                          ,tweet_var=self.tweet,label_var=self.datalabel,
                                            filepath_var=self.filepath,k_var=self.k,kfold_var=self.kfold,paramknn_var=self.params_knn) # create window if its None or destroyed
            self.knn_popup.title(" Klasifikasi  K-Nearest Neighbors") 
        else:
            self.knn_popup.focus()
    def get_entry_label(self):
        return self.datalabel.get()
    def get_entry_tweet(self):
        return self.tweet.get()
    def get_entry_clean(self):
        return self.cleantweet.get()
    
    
