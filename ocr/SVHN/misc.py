import pickle


def save_obj(obj, filename):
    '''
    Save an object in pickle format.
    Parameters
    ----------
    obj : object
        Object to save.
    filename: str
        path/filename to saved the object.
    '''
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    '''
    Load an object from pickle format.
    Parameters
    ----------
    filename: str
        path/filename of the saved object.
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)
