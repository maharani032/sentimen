import os
import threading
import customtkinter
from tkinter import filedialog, ttk
import pandas as pd
from tkinter import messagebox
from dotenv import load_dotenv
import tweepy


class CrawlingFrame(customtkinter.CTkToplevel):

    def is_numeric(char):
        # """Validasi apakah input adalah numerik"""
        return char.isdigit()
    def __init__(self,master):
        super().__init__(master)
        self.search=customtkinter.StringVar()
        self.limit=customtkinter.StringVar()
        
        # self.filepath.trace_add('write', lambda *args: self.Load_excel_data(self.filepath.get()))
        # self.header = customtkinter.CTkLabel(self, text="Crawling Data")
        # self.header.grid(row=0, column=0, padx=10,sticky='ew',columnspan=2,pady=20)
        self.header = customtkinter.CTkLabel(self, text="Crawling Data",font=customtkinter.CTkFont(weight='bold'))
        self.header.grid(row=0, column=0, padx=10,sticky='ew',columnspan=2,pady=10)
        self.label_search=customtkinter.CTkLabel(self,text='Search :')
        self.label_search.grid(row=1, column=0,sticky='w',padx=10,pady=4)
        self.entry_search = customtkinter.CTkEntry(self,textvariable=self.search,width=200)
        self.entry_search.grid(row=1, column=1,pady=4,padx=10,sticky='w')

        # # label Limit
        self.label_limit=customtkinter.CTkLabel(self,text='Limit :')
        self.label_limit.grid(row=2, column=0,sticky='w',padx=10,pady=4)
        self.entry_limit = customtkinter.CTkEntry(self,textvariable=self.limit,width=200)
        self.entry_limit.grid(row=2, column=1,padx=10,pady=4,sticky='w')
        self.entry_limit['validate'] = 'key'
        self.entry_limit['validatecommand'] = (self.entry_limit.register(self.is_numeric), '%P')
        
        crawling_data_button=customtkinter.CTkButton(self,text='Start',command=lambda:threading.Thread(
            target=self.CrawlingData, args=(self.entry_search.get(),self.entry_limit.get())).start())
        crawling_data_button.grid(row=3,column=0,columnspan=2,pady=20,padx=20)
    def CrawlingData(self,search,limit):
        print(search,limit)

        load_dotenv()
        self.consumer_key = os.getenv('consumer_key')
        self.consumer_secret = os.getenv('consumer_secret')
        self.access_token = os.getenv('access_token')
        self.access_token_secret = os.getenv('access_token_secret')

        try:
            auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
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
                    db_tweets.to_csv(file.name, index=False)
                else:
                    db_tweets.to_excel(file.name, index=False)
            self.destroy()
        except tweepy.errors.Unauthorized as e:
            print(e)
            self.destroy()