import os
import errno


def maybe_create_folder(folder: str):
    # create folder if it doesn't exist
    try:
        os.makedirs(folder)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
