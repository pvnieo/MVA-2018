# stdlib
import os
import pickle


DUMPS = "./dumps/"


def is_result_saved(filename):
        """Check if filename exists
        """
        filename = os.path.join(DUMPS, filename)

        return os.path.exists(filename)


def save_result(filename, obj):
    """ Dump a pickle of the result in the cachefile
    """
    filename = os.path.join(DUMPS, filename)

    with open(filename, "wb") as cache:
        pickle.dump(obj, cache)


def get_result(filename):
    """ Load pickled result from filename
    """
    filename = os.path.join(DUMPS, filename)
    with open(filename, "rb") as cache:
        try:
            return pickle.load(cache)
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError("Error while pickling!")
            print(e)
    return None
