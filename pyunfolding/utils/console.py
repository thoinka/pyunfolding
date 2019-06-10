def in_ipython_frontend():
    """
    check if we're inside an an IPython zmq frontend
    Note: Stolen from Pandas
    """
    try:
        ip = get_ipython()  # noqa
        return 'zmq' in str(type(ip)).lower()
    except NameError:
        pass

    return False