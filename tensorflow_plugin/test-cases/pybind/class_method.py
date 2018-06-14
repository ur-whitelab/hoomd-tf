import _class_method

class ClassMethod:
    def __init__(self):
        self.cpp_ref = _class_method.ClassMethod(self)

    def test(self):
        print('hello from python!')