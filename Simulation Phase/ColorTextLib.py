__all__ = ['bcolors','printC','colorText']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printC(text,color):
    print(colorText(text,color))

def colorText(text,color):
    #ritorna una stringa in formato "colorato"
    return color + text + bcolors.ENDC

