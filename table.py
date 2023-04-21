import customtkinter
from tkinter import ttk
import pandas as pd
from tkinter import messagebox


class tableFrame(customtkinter.CTkFrame):
    def clear_data(self):
        self.tree= self.children['!treeview']
        self.tree.delete(*self.tree.get_children())
        return None
    def Load_excel_data(self,filepath):
        try:
            excel_filename = r"{}".format(filepath)
            if excel_filename[-4:] == ".csv":
                df = pd.read_csv(excel_filename)
            else:
                df = pd.read_excel(excel_filename)
            print(excel_filename)
        except ValueError:
            messagebox.showerror("Information", "The file you have chosen is invalid")
            return None
        except FileNotFoundError:
            messagebox.showerror("Information", f"No such file as {filepath}")
            return None
        self.clear_data()
        self.tree["column"] = list(df.columns)
        # self.tree["show"] = "headings"
        for column in self.tree["columns"]:
            self.tree.heading(column, text=column,anchor='e')
            self.tree.column(column,anchor='e', stretch=False)

        df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
        for row in df_rows:
            self.tree.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
        hs=ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        hs.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=hs.set)
        ttk.Style().configure("TreeviewItem", rowheight = 50, font = (None, 50))
        vs=ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        vs.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.grid(row=0, column=0, sticky="nsew",padx=0,pady=0)
        return None    
    
    def __init__(self,master,header_name,filepath_var=None):
        super().__init__(master)
        
        self.header_name = header_name
        self.filepath = filepath_var
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.filepath.trace_add('write', lambda *args: self.Load_excel_data(self.filepath.get()))
        # self.header = customtkinter.CTkLabel(self, text=self.header_name,width=320)
        # self.header.grid(row=0, column=0, padx=10,sticky='n')
        
        self.tree=ttk.Treeview(self,selectmode='extended')
        ttk.Style().configure("TreeviewItem", rowheight = 35)
        # self.tree.grid(row=2,column=0)
        # self.columnconfigure(0, weight=1) # column with treeview
        # self.rowconfigure(2, weight=1) 

        # row with treeview  
        