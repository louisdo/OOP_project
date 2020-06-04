from lib import Utils
from train_infer import TransformerInference
from user_interface import Interface
from tkinter import Tk


class Main:
    def __init__(self, device: str = "cpu"):
        self.CONFIG = Utils.get_config_yml("./config/main.yml")

        inference = TransformerInference(checkpoint_folder = self.CONFIG["checkpoint_folder"],
                                         vocab_folder = self.CONFIG["vocab_folder"],
                                         device = device)

        self.root = Tk()
        self.root.geometry("600x100+400+200")

        self.app = Interface(self.root, inference)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    main = Main()
    main.run()