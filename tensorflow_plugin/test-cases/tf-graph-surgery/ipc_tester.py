import _ipc_tester

class IpcTester:
    def __init__(self):
        self.cpp_ref = _ipc_tester.IpcTester(self)

    def test(self):
        print('hello from python!')