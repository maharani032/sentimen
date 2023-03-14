from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import os
    
def browseFiles():
    filename = filedialog.askopenfilename(
        initialdir = "/",
        title = "Select a File xlsv or csv",
        filetypes = (
            ("Excel file","*.xlsx*"),
            ("CSV file","*.csv*")
            )
    ) 
    if os.path.splitext(filename)[1].lower() == '.csv':
        try:
            data=pd.read_csv(filename)
            data.info()
        except:
            messagebox.showerror('python error','data tidak bisa terbaca')

        
    elif os.path.splitext(filename)[1].lower() == '.xlsx':
        try:
            data= pd.read_excel(filename)
            data.info()
        except:
            messagebox.showerror('python error','data tidak bisa terbaca')

def closeapp():
    root.destroy()

root =Tk()

root.title('Analisis Sentimen dengan NBC dan KNN')
root.geometry('400x400')


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=browseFiles)
filemenu.add_command(label="Close", command=closeapp)
menubar.add_cascade(label="File", menu=filemenu)



root.config(menu=menubar)
root.mainloop()