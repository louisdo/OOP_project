import tkinter.messagebox as msbox
from tkinter import *
from tkinter.ttk import Frame, Label, Entry


class Interface(Frame):
    def __init__(self, parent, process_function):
        Frame.__init__(self, parent)
        self.input_sentence = " "
        self.entry = 0
        self.parent = parent
        self.input_bar()
        self.function_button()
        self.process_function = process_function

    def input_bar(self):
        self.parent.title("Result")
        self.pack(fill=BOTH, expand=True)

        frame = Frame(self)
        frame.pack(fill=BOTH, expand=True)

        label = Label(frame, text="Input", width=6)
        label.pack(side=LEFT, anchor= W, padx=5, pady=5)

        self.entry = Entry(frame)
        self.entry.pack(fill=BOTH, padx=5, pady=5, expand= True)

    def function_button(self):
        frame_button = Frame(self, relief=RAISED, borderwidth=1)
        frame_button.pack(fill=BOTH, expand = True)

        self.pack(fill=BOTH, expand=True)
        close_button = Button(self, text="Close", command=self.quit)
        close_button.pack(side=RIGHT, padx=5, pady=5)

        ok_button = Button(self, text="OK", command=self.show_result)
        ok_button.pack(side=RIGHT)

    def show_result(self):
        self.input_sentence = self.entry.get()
        try:
            result = self.process_function(self.input_sentence)
            msbox.showinfo(title="RESULT", message=result)
        except Exception:
            msbox.showinfo(title="ERROR", message="Something went wrong! Please try again")

