import pickle


def save_v(file_folder, file_object):
    file_obj = open(file_folder, 'w+')
    pickle.dump(file_object, file_obj)


def save_v(file_folder):
    file_obj = open(file_folder, 'r')
    return pickle.load(file_obj)


save_v('123', [1,2,3,4])