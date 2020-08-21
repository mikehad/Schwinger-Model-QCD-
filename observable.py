"""
Observable class
"""

__all__ = [
    "History",
    "history",
]


from functools import wraps
import numpy as np
import matplotlib.pyplot as plt



class History(np.ndarray):
    """
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        print('In __array_finalize__:')
        print('   self is %s' % repr(self))
        print('   obj is %s' % repr(obj))
        if obj is None: return
        self.info = getattr(obj, 'info', None)
    """
    def __array_wrap__(self, out_arr, context=None):
        out_arr.therm_time = self.therm_time
        return super().__array_wrap__(self, out_arr, context)

    @property
    def tau_int(self):
        return np.sum(self.autocorellation())

    def autocorellation(self):
        self=np.array(self)
        n = len(self)
        f = np.fft.fft(self - self.mean(), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[:n].real
        acf /= 4 * n
        acf /= acf[0]
        return acf

    @property
    def therm_time(self):
        "Returns the "
        return getattr(self, "_therm_time", self.shape[0] // 10)

    @therm_time.setter
    def therm_time(self, value):
        self._therm_time=value

    @property
    def thermalized(self):
        return self[self.therm_time :].view(np.ndarray)

    def mean(self):
        return self.thermalized.mean()

    def std(self):
        return self.thermalized.std()

    def __str__(self):
        return "value(err)"

    def variance(self):
        return self.thermalized.var()

    def autocorellation_plot(self):
        plt.plot(self.autocorellation()) #label = str(T[i]))
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel("iterations")
        plt.ylabel("autocorellations")
        plt.show()

    def observable_plot(self):
        plt.plot(self) #label = str(T[i]))
        plt.axhline(y=self.mean(), color='r', linestyle='-')
        plt.xlabel("iterations")
        plt.ylabel("observable")
        plt.show()


@wraps(np.array)
def history(*args, **kwargs):
    return np.array(*args, **kwargs).view(History)
