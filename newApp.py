
import os
import tkinter
from tkinter import filedialog
import customtkinter
from crawling import CrawlingFrame
from datainfo import DataInfoFrame
import pandas as pd
from tkinter import messagebox
from table import tableFrame




class App(customtkinter.CTk):
    
    def browseFiles(self):
        filename = filedialog.askopenfilename(
            initialdir = "/",
            title = "Select a File xlsv or csv",
            filetypes = (
                ("Excel file","*.xlsx*"),
                ("CSV file","*.csv*")
                )
        ) 
        try:
            if os.path.splitext(filename)[1].lower() == '.csv':
                data=pd.read_csv(filename)
            elif os.path.splitext(filename)[1].lower() == '.xlsx':
                data= pd.read_excel(filename)
            jumlah_per_kolom = data.count()

            self.filepath.set(filename)
            self.filename.set(os.path.basename(filename))        
            self.datafile.set(' '.join([f'{kolom}: {jumlah_per_kolom[kolom]} ' for kolom in jumlah_per_kolom.index]))
            list_kolom = data.columns.to_list()
            string_kolom = ', '.join(list_kolom)
            self.listcolumn.set(string_kolom)

        except ValueError:
            messagebox.showerror("Information","File rusak")
        #     return None
        except FileNotFoundError:
            messagebox.showerror("Information", f"No such file as {filename}")
            # return None
    def closeapp(self):
        return self.destroy()
    def __init__(self):
        super().__init__()

        self.title("Analisis Sentimen")

        width= self.winfo_screenwidth()
        height= self.winfo_screenheight()
        #setting tkinter window size
        self.geometry("%dx%d" % (width, height))
        # self.geometry("720x480")
        # window.attributes('-fullscreen', True)

        # self.grid_rowconfigure(1, weight=1)
        # self.columnconfigure(2, weight=1)
        self.filename = customtkinter.StringVar()
        self.filepath=customtkinter.StringVar()
        self.datafile=customtkinter.StringVar()

        self.tweet=customtkinter.StringVar()
        self.label=customtkinter.StringVar()
        self.cleantweet=customtkinter.StringVar()

        self.listcolumn=customtkinter.StringVar()

        menubar = tkinter.Menu()
        filemenu = tkinter.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.browseFiles)
        filemenu.add_command(label="Close", command=self.closeapp)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)
        self.grid_columnconfigure(1, weight = 1)
        self.grid_rowconfigure(0, weight = 1)

        # self.data_info=customtkinter.CTkFrame(master=self,height=200,width=300)

        self.data_info=DataInfoFrame(master=self,header_name="Data Info"
                                     ,filename_var=self.filename,filepath_var=self.filepath
                                     ,datafile_var=self.datafile,listcolumn_var=self.listcolumn)
        self.data_info.grid(row=0,column=0,padx=10,pady=10,sticky='nsew')
        # self.data_info.place()

        

        self.data_tabel=tableFrame(master=self,header_name="Data",filepath_var=self.filepath)
        self.data_tabel.grid(row=0,column=1,padx=10,pady=10,sticky='NSEW')

        
        

    
        return
if __name__ == "__main__":
    app = App()
    app.mainloop()
