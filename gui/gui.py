import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from tkinter import IntVar, Checkbutton
import subprocess
import os


class CSVFileSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV File Selector")
        self.train_file_path = None
        self.test_file_path = None
        self.features = []
        self.target_variable = None
        self.output_file_path = None
        self.genome_file_path = ""

        # UI components
        self.train_label = tk.Label(self, text="Select Training CSV:")
        self.train_label.grid(row=0, column=0, padx=10, pady=10)

        self.train_entry_var = tk.StringVar()
        self.train_entry = tk.Entry(
            self, textvariable=self.train_entry_var, state="disabled", width=50
        )
        self.train_entry.grid(row=0, column=1, padx=10, pady=10)

        self.train_browse_button = tk.Button(
            self, text="Browse", command=self.browse_train_file
        )
        self.train_browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.test_label = tk.Label(self, text="Select Test CSV:")
        self.test_label.grid(row=1, column=0, padx=10, pady=10)

        self.test_entry_var = tk.StringVar()
        self.test_entry = tk.Entry(
            self, textvariable=self.test_entry_var, state="disabled", width=50
        )
        self.test_entry.grid(row=1, column=1, padx=10, pady=10)

        self.test_browse_button = tk.Button(
            self, text="Browse", command=self.browse_test_file
        )
        self.test_browse_button.grid(row=1, column=2, padx=10, pady=10)

        self.column_listbox = tk.Listbox(
            self, selectmode=tk.MULTIPLE, exportselection=0, height=10
        )

        self.output_label = tk.Label(self, text="Select Output Directory:")
        self.output_label.grid(row=3, column=0, padx=10, pady=10)

        self.output_entry_var = tk.StringVar()
        self.output_entry = tk.Entry(
            self, textvariable=self.output_entry_var, state="disabled", width=50
        )
        self.output_entry.grid(row=3, column=1, padx=10, pady=10)

        self.output_browse_button = tk.Button(
            self, text="Browse", command=self.browse_output_directory
        )
        self.output_browse_button.grid(row=3, column=2, padx=10, pady=10)

        self.number_threads_label = tk.Label(self, text="Number of Threads:")
        self.number_threads_label.grid(row=4, column=0, padx=10, pady=5, sticky="e")
        self.number_threads_var = tk.StringVar(value="9")
        self.number_threads_entry = tk.Entry(self, textvariable=self.number_threads_var)
        self.number_threads_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        self.node_types_label = tk.Label(self, text="Possible Node Types:")
        self.node_types_label.grid(row=4, column=2, padx=10, pady=5)
        self.node_types_options = ["simple", "UGRNN", "MGU", "GRU", "delta", "LSTM"]
        self.node_type_vars = [IntVar(value=1) for _ in self.node_types_options]

        for i, node_type in enumerate(self.node_types_options):
            checkbox = Checkbutton(
                self,
                text=node_type,
                variable=self.node_type_vars[i],
                onvalue=1,
                offvalue=0,
            )
            if i < 3:
                checkbox.grid(row=4, column=i + 3, padx=5, pady=5)
            else:
                checkbox.grid(row=5, column=i, padx=5, pady=5)

        self.time_offset_label = tk.Label(self, text="Time Offset:")
        self.time_offset_label.grid(row=5, column=0, padx=10, pady=5, sticky="e")
        self.time_offset_var = tk.StringVar(value="1")
        self.time_offset_entry = tk.Entry(self, textvariable=self.time_offset_var)
        self.time_offset_entry.grid(row=5, column=1, padx=10, pady=5, sticky="w")

        self.number_islands_label = tk.Label(self, text="Number of Islands:")
        self.number_islands_label.grid(row=6, column=0, padx=10, pady=5, sticky="e")
        self.number_islands_var = tk.StringVar(value=10)
        self.number_islands_entry = tk.Entry(self, textvariable=self.number_islands_var)
        self.number_islands_entry.grid(row=6, column=1, padx=10, pady=5, sticky="w")

        self.std_message_level_label = tk.Label(self, text="Standard Message Level:")
        self.std_message_level_label.grid(row=6, column=2, padx=10, pady=5)

        self.log_levels = ["NONE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.std_message_level_var = (
            tk.StringVar()
        )  # Create StringVar without initial value

        std_message_level_menu = tk.OptionMenu(
            self, self.std_message_level_var, *self.log_levels
        )
        std_message_level_menu.grid(row=6, column=3, padx=5, pady=5)

        # Set the default value after creating the OptionMenu
        self.std_message_level_var.set("INFO")

        self.file_message_level_label = tk.Label(self, text="File Message Level:")
        self.file_message_level_label.grid(row=7, column=2, padx=10, pady=5)

        self.file_message_level_var = tk.StringVar()
        file_message_level_menu = tk.OptionMenu(
            self, self.file_message_level_var, *self.log_levels
        )
        file_message_level_menu.grid(row=7, column=3, padx=5, pady=5)

        self.file_message_level_var.set("NONE")

        self.size_islands_label = tk.Label(self, text="Island Size:")
        self.size_islands_label.grid(row=7, column=0, padx=10, pady=5, sticky="e")
        self.size_islands_var = tk.StringVar(value=10)
        self.size_islands_entry = tk.Entry(self, textvariable=self.size_islands_var)
        self.size_islands_entry.grid(row=7, column=1, padx=10, pady=5, sticky="w")

        self.max_genomes_label = tk.Label(self, text="Max Genomes:")
        self.max_genomes_label.grid(row=8, column=0, padx=10, pady=5, sticky="e")
        self.max_genomes_var = tk.StringVar(value=2000)
        self.max_genomes_entry = tk.Entry(self, textvariable=self.max_genomes_var)
        self.max_genomes_entry.grid(row=8, column=1, padx=10, pady=5, sticky="w")

        self.bp_iter_label = tk.Label(self, text="BP Iterations:")
        self.bp_iter_label.grid(row=9, column=0, padx=10, pady=5, sticky="e")
        self.bp_iter_var = tk.StringVar(value=10)
        self.bp_iter_entry = tk.Entry(self, textvariable=self.bp_iter_var)
        self.bp_iter_entry.grid(row=9, column=1, padx=10, pady=5, sticky="w")

        self.select_features_button = tk.Button(
            self, text="Select Features", command=self.select_features
        )
        self.select_features_button.grid(row=10, column=1, columnspan=3, pady=10)

        self.Initiate_training_button = tk.Button(
            self, text="Train EXAMM", command=self.Initiate_training, state="disabled"
        )
        self.Initiate_training_button.grid(row=11, column=1, columnspan=3, pady=10)

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
        self.features = [
            self.column_listbox.get(index)
            for index in range(self.column_listbox.size())
        ]

        feature_selection_window = FeatureSelectionWindow(self, self.features)
        self.wait_window(feature_selection_window)

    def browse_output_directory(self):
        directory_path = filedialog.askdirectory()
        self.output_file_path = directory_path
        self.output_entry_var.set(directory_path)

    def Initiate_training(self):
        command = (
            f"build/multithreaded/examm_mt"
            f" --number_threads {self.number_threads_var.get()} "
            f'--training_filenames "{self.train_file_path}" '
            f'--test_filenames "{self.test_file_path}" '
            f"--time_offset {self.time_offset_var.get()} "
            f'--input_parameter_names {" ".join(self.features)} '
            f"--output_parameter_names {self.target_variable} "
            f"--number_islands {self.number_islands_var.get()} "
            f"--island_size {self.size_islands_var.get()} "
            f"--max_genomes {self.max_genomes_var.get()} "
            f"--bp_iterations {self.bp_iter_var.get()} "
            f'--output_directory "{self.output_file_path}" '
            f'--possible_node_types {" ".join([node_type for node_type, var in zip(self.node_types_options, self.node_type_vars) if var.get()])} '
            f"--std_message_level {self.std_message_level_var.get()} "
            f"--file_message_level {self.file_message_level_var.get()} "
        )

        print(f'--output_directory "{self.output_file_path}" ')
        try:
            # Run the command
            subprocess.run(command, shell=True)
            messagebox.showinfo("Run Information", "Training completed successfully!")

            os.chdir(self.output_file_path)

            for filename in os.listdir():
                if filename.startswith("global"):
                    if filename.endswith(".bin"):
                        self.genome_file_path = os.path.abspath(filename)
                        break

            if self.genome_file_path == "":
                messagebox.showwarning(
                    "File Not Found",
                    "No file matching 'global*.bin' found in the output directory.",
                )
            else:
                messagebox.showinfo("Code Prediction Generations :)")

        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Error executing command: {e}")


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

        self.feature_listbox = tk.Listbox(
            self, selectmode=tk.MULTIPLE, exportselection=0, height=10
        )
        for feature in self.features:
            self.feature_listbox.insert(tk.END, feature)
        self.feature_listbox.grid(row=0, column=1, padx=10, pady=10)

        self.target_label = tk.Label(self, text="Select Target Variable:")
        self.target_label.grid(row=1, column=0, padx=10, pady=10)

        self.target_listbox = tk.Listbox(
            self, selectmode=tk.SINGLE, exportselection=0, height=5
        )
        for feature in self.features:
            self.target_listbox.insert(tk.END, feature)
        self.target_listbox.grid(row=1, column=1, padx=10, pady=10)

        self.confirm_button = tk.Button(
            self, text="Confirm", command=self.confirm_selection
        )
        self.confirm_button.grid(row=2, column=0, columnspan=2, pady=10)

    def confirm_selection(self):
        selected_features = self.feature_listbox.curselection()
        selected_target = self.target_listbox.curselection()
        if not selected_target:
            messagebox.showwarning("Warning", "Please select the target variable.")
            return

        self.parent.features = [
            self.feature_listbox.get(index) for index in selected_features
        ]
        self.parent.target_variable = self.target_listbox.get(selected_target[0])

        self.parent.Initiate_training_button[
            "state"
        ] = "active"  # Enable Train EXAMM button in the main window
        self.destroy()


if __name__ == "__main__":
    app = CSVFileSelector()
    app.mainloop()
