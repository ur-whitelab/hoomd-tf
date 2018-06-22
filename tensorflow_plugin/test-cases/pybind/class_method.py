import _class_method

class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
class Singleton(_Singleton('SingletonMeta', (object,), {})): pass

class ClassMethod(Singleton):
    def __init__(self):
        self.cpp_ref = _class_method.ClassMethod(self)

    def test(self):
        print('hello from python!')