from cgitb import text
from enum import auto
from logging import error
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import tkinter
from tkinter.font import Font
from turtle import width
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
    try:
        if os.path.splitext(filename)[1].lower() == '.csv':
            data=pd.read_csv(filename)
        elif os.path.splitext(filename)[1].lower() == '.xlsx':
            data= pd.read_excel(filename)
        jumlah_per_kolom = data.count()
        filePath.set(filename)
        fileName.set(os.path.basename(filename))        
        file_column.set(' '.join([f'{kolom}: {jumlah_per_kolom[kolom]} ' for kolom in jumlah_per_kolom.index]))
        Load_excel_data(filePath=filename)
        return None
    except ValueError:
        messagebox.showerror("Information", "The file you have chosen is invalid")
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
        tree.column(column, width=40,stretch=False)
    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tree.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert

# scrollbar
    hs=Scrollbar(tableFrame,orient=HORIZONTAL,command=tree.xview)
    tree.configure(xscrollcommand=hs.set)
    hs.grid(row=0,column=1,sticky='ew')
    # hs.pack(side='BOTTOM',fill=X)
    return None
def handle_click(event):
    if tree.identify_region(event.x, event.y) == "separator":
        return "break"

def clear_data():
    tree.delete(*tree.get_children())
    return None

root =Tk()
filePath= StringVar()
fileName=StringVar()
file_column=StringVar()

root.title('Analisis Sentimen dengan NBC dan KNN')
root.geometry('500x800')
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0) # makes the root window fixed in size.
MainFrame = Frame(root, width=500, height=500, relief='raised', borderwidth=5)
# MainFrame.pack(expand=True, fill='both')
innerFrame = Frame(MainFrame, borderwidth=3,background='red',width=500)
# innerFrame.pack(fill="both", expand=True)
tableFrame = Frame(MainFrame, borderwidth=3,width=500,height=200)
buttonFrame= Frame(MainFrame, borderwidth=3,width=500,height=200)
# tableFrame.pack(fill="both", expand=True)

for frame in [MainFrame, innerFrame, tableFrame,buttonFrame]:
    frame.pack(expand=True, fill='both')
    frame.pack_propagate(0)

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


# Create TreeView
tree = ttk.Treeview(tableFrame,  selectmode="extended")
tree.pack()
# treescrolly = Scrollbar(tableFrame, orient="vertical")
# treescrollx = Scrollbar(tableFrame, orient="horizontal")
# # treescrolly = Scrollbar(tableFrame, orient="vertical", command=tree.yview)
# # treescrollx = Scrollbar(tableFrame, orient="horizontal", command=tree.xview)
# tree.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
# tree.configure(selectmode='extended')
# treescrollx.configure(command=tree.xview)
# treescrolly.configure(command=tree.yview)

# treescrollx.place(relx=0.002,rely=0.5,width=500,height=22)
# treescrollx.pack(side="bottom", fill="x")
# treescrolly.pack(side="right", fill="y")

# tree_scroll = Scrollbar(tableFrame)
# tree_scroll.pack(side=RIGHT, fill=Y)

# tree_scroll.config(command=my_tree.yview)
root.config(menu=menubar)
root.mainloop()