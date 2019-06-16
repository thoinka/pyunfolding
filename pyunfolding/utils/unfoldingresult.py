from ..plot import plot_unfolding_result


class UnfoldingResult:
    def __init__(self, f, f_err, success, *args, **kwargs):
        self.f = f
        self.f_err = f_err
        self.success = success
        for key, val in kwargs.items():
            self.__dict__[key] = val

    def __str__(self):
        if self.success:
            s = "Successful unfolding:\n"
        else:
            s = "Unsuccessful unfolding:\n"
        for i, (f, ferr1, ferr2) in enumerate(zip(self.f, self.f_err[0], self.f_err[1])):
            s += "Var {}:\t{:.2f} +{:.2f} -{:.2f}\n".format(i + 1,
                                                            f, ferr1, ferr2)
        try:
            s += "Error-Message: {}".format(self.error)
        except:
            s += "No error message was left."
        return s

    def plot(self, *args, **kwargs):
        return plot_unfolding_result(self, *args, **kwargs)

    def __repr__(self):
        return self.__str__()

    def update(self, **kwargs):
        self.__dict__.update(kwargs)