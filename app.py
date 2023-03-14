from cgitb import text
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
            jumlah_per_kolom = data.count()
            filePath.set(filename)
            fileName.set(os.path.basename(filename))
            file_column.set(' '.join([f'{kolom}: {jumlah_per_kolom[kolom]} ' for kolom in jumlah_per_kolom.index]))
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
filePath= StringVar()
fileName=StringVar()
file_column=StringVar()

root.title('Analisis Sentimen dengan NBC dan KNN')
root.geometry('500x500')
innerFrame = Frame(root, borderwidth=3)
innerFrame.pack(fill="both", expand=True)

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=browseFiles)
filemenu.add_command(label="Close", command=closeapp)
menubar.add_cascade(label="File", menu=filemenu)
# font
my_font1=('times', 10, 'normal')
# label file
label_f_path=Label(innerFrame,text='File path   :',font=my_font1)
label_f_path.grid(row=0, column=0 )

# label filename
label_f_name=Label(innerFrame,text='File name   :',font=my_font1)
label_f_name.grid(row=1, column=0 ,pady=4)

# label c_table
label_c_table=Label(innerFrame,text='Data Table   :',font=my_font1)
label_c_table.grid(row=2, column=0 ,pady=4)

entry_file_path = Entry(innerFrame,font=my_font1,textvariable=filePath,width=50)
entry_file_path.grid(row=0, column=1,padx=10)
entry_file_path.config(state= "disabled")

entry_f_name = Entry(innerFrame,font=my_font1,textvariable=fileName,width=50)
entry_f_name.grid(row=1, column=1,padx=10,pady=4)
entry_f_name.config(state= "disabled")

entry_f_name = Entry(innerFrame,font=my_font1,textvariable=file_column,width=50)
entry_f_name.grid(row=2, column=1,padx=10,pady=4)
entry_f_name.config(state= "disabled")


root.config(menu=menubar)
root.mainloop()