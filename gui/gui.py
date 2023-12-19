import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

class CSVFileSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV File Selector")
        self.train_file_path = None
        self.test_file_path = None
        self.features = []
        self.target_variable = None
        self.output_file_path = None

        # UI components
        self.train_label = tk.Label(self, text="Select Training CSV:")
        self.train_label.grid(row=0, column=0, padx=10, pady=10)

        self.train_entry_var = tk.StringVar()
        self.train_entry = tk.Entry(self, textvariable=self.train_entry_var, state="disabled", width=50)
        self.train_entry.grid(row=0, column=1, padx=10, pady=10)

        self.train_browse_button = tk.Button(self, text="Browse", command=self.browse_train_file)
        self.train_browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.test_label = tk.Label(self, text="Select Test CSV:")
        self.test_label.grid(row=1, column=0, padx=10, pady=10)

        self.test_entry_var = tk.StringVar()
        self.test_entry = tk.Entry(self, textvariable=self.test_entry_var, state="disabled", width=50)
        self.test_entry.grid(row=1, column=1, padx=10, pady=10)

        self.test_browse_button = tk.Button(self, text="Browse", command=self.browse_test_file)
        self.test_browse_button.grid(row=1, column=2, padx=10, pady=10)

        
        self.column_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, exportselection=0, height=10)
        self.column_listbox.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
        
        self.select_features_button = tk.Button(self, text="Select Features", command=self.select_features)
        self.select_features_button.grid(row=4, column=0, columnspan=3, pady=10)

        self.initiate_run_button = tk.Button(self, text="Initiate Run", command=self.initiate_run, state="disabled")
        self.initiate_run_button.grid(row=5, column=0, columnspan=3, pady=10)

    def browse_train_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.train_file_path = file_path
        self.train_entry_var.set(file_path)
        self.load_train_columns()
        

    def browse_test_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.test_file_path = file_path
        self.test_entry_var.set(file_path)

    def load_train_columns(self):
        if self.train_file_path:
            try:
                df = pd.read_csv(self.train_file_path)
                columns = df.columns.tolist()
                # Add columns to the listbox without displaying it on the GUI
                for column in columns:
                    self.column_listbox.insert(tk.END, column)
            except Exception as e:
                    messagebox.showerror("Error", f"Error loading columns: {e}")
        else:
            messagebox.showwarning("Warning", "Please select a training CSV file.")


    def select_features(self):
        self.features = [self.column_listbox.get(index) for index in range(self.column_listbox.size())]

        feature_selection_window = FeatureSelectionWindow(self, self.features)
        self.wait_window(feature_selection_window)

    def initiate_run(self):
        result_text = (
            f"Training File: {self.train_file_path}\n"
            f"Test File: {self.test_file_path}\n"
            f"Selected Features: {', '.join(self.features)}\n"
            f"Target Variable: {self.target_variable}\n"
            f"Output File: {self.output_file_path}"
        )
        messagebox.showinfo("Run Information", result_text)

class FeatureSelectionWindow(tk.Toplevel):
    def __init__(self, parent, features):
        super().__init__(parent)
        self.title("Feature Selection")
        self.parent = parent
        self.features = features
        self.target_variable = None

        # UI components
        self.feature_label = tk.Label(self, text="Select Features:")
        self.feature_label.grid(row=0, column=0, padx=10, pady=10)

        self.feature_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, exportselection=0, height=10)
        for feature in self.features:
            self.feature_listbox.insert(tk.END, feature)
        self.feature_listbox.grid(row=0, column=1, padx=10, pady=10)

        self.target_label = tk.Label(self, text="Select Target Variable:")
        self.target_label.grid(row=1, column=0, padx=10, pady=10)

        self.target_listbox = tk.Listbox(self, selectmode=tk.SINGLE, exportselection=0, height=5)
        for feature in self.features:
            self.target_listbox.insert(tk.END, feature)
        self.target_listbox.grid(row=1, column=1, padx=10, pady=10)

        self.confirm_button = tk.Button(self, text="Confirm", command=self.confirm_selection)
        self.confirm_button.grid(row=2, column=0, columnspan=2, pady=10)

    def confirm_selection(self):
        selected_features = self.feature_listbox.curselection()
        selected_target = self.target_listbox.curselection()
        if not selected_target:
            messagebox.showwarning("Warning", "Please select the target variable.")
            return

        self.parent.features = [self.feature_listbox.get(index) for index in selected_features]
        self.parent.target_variable = self.target_listbox.get(selected_target[0])

        self.parent.initiate_run_button["state"] = "active"  # Enable Initiate Run button in the main window
        self.destroy()

if __name__ == "__main__":
    app = CSVFileSelector()
    app.mainloop()
