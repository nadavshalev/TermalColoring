import pickle

def save_v(filename1 ,object1):
    filehandler = open(filename1, 'wb')
    pickle.dump(object1, filehandler)

def open_v(filename):
    filehandler = open(filename, 'rb')
    object = pickle.load(filehandler)
    return object
