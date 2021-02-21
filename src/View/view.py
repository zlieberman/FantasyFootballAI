import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MODEL_OPTIONS = ["1", "2", "3", "4", "5"]
LOSS_OPTIONS = ["A", "B", "C", "D"]

# root = Tk()
# Control Frame
#   - Buttons
#   - Text Boxes
# Visual Frame
#   - Training Canvas
#   - Testing Canvas

class View():
    def __init__(self):
        self.root = tk.Tk()
        self.app = MainFrame(master=self.root)
        self.app.mainloop()


# frame containing both the visual and control frames
class MainFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.control_frame = ControlFrame(self.master)
        self.visual_frame = VisualFrame(self.master)
    

# frame containing both the testing and training control frames
class ControlFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(side="bottom")
        self.train_frame = TrainControls(self)
        self.test_frame = TestControls(self)


# frame containing the controls to train the model
class TrainControls(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(side="left", expand=False, fill="both")
        self.create_widgets()

    def create_widgets(self):
        # create text inputs (Entry objects)
        epoch_label = tk.Label(self, text="Epochs:")
        epoch_label.grid(row=0,column=0)
        self.epoch_entry = tk.Entry(self)
        self.epoch_entry.grid(row=0, column=1)

        # create drop down menus (tk.OptionMenu)
        # model type 
        model_type_label = tk.Label(self, text="Model Type:")
        model_type_label.grid(row=1,column=0)
        self.model_type_variable = tk.StringVar(self)
        self.model_type_variable.set(MODEL_OPTIONS[0]) # default value
        self.model_type_menu = tk.OptionMenu(self, self.model_type_variable, *MODEL_OPTIONS)
        self.model_type_menu.grid(row=1, column=1)

        # loss function
        loss_fn_label = tk.Label(self, text="Loss Function:")
        loss_fn_label.grid(row=2,column=0)
        self.loss_fn_variable = tk.StringVar(self)
        self.loss_fn_variable.set(LOSS_OPTIONS[0]) # default value
        self.loss_fn_menu = tk.OptionMenu(self, self.loss_fn_variable, *LOSS_OPTIONS)
        self.loss_fn_menu.grid(row=2, column=1)

        # create buttons (tk.Button)
        self.run_button = tk.Button(self, text="Train", command=self.run)
        self.run_button.grid(row=0, column=2)

        self.stop_button = tk.Button(self, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=2)

    def run(self):
        print("Train")

    def stop(self):
        print("stop")


# frame containing the controls to test the model
class TestControls(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(side="right", expand="False", fill="both")
        self.create_widgets()

    def create_widgets(self):
        # create buttons (tk.Button)
        self.run_button = tk.Button(self, text="Test", command=self.run)
        self.run_button.grid(row=1, column=0)

        self.stop_button = tk.Button(self, text="Stop", command=self.stop)
        self.stop_button.grid(row=2, column=0)

    def run(self):
        print("Test")

    def stop(self):
        print("stop")



# frame to hold the visual representations of the data
class VisualFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(side="top")
        self.create_canvas()

    def create_canvas(self):
        # create the training graph
        train_figure = plt.Figure(figsize=(6,5), dpi=100)
        train_ax = train_figure.add_subplot(111)
        train_plot = FigureCanvasTkAgg(train_figure, self)
        train_plot.get_tk_widget().grid(row=0,column=0)

        # create the testing graph
        test_figure = plt.Figure(figsize=(6,5), dpi=100)
        test_ax = test_figure.add_subplot(111)
        test_plot = FigureCanvasTkAgg(test_figure, self)
        test_plot.get_tk_widget().grid(row=0,column=1)



view = View()