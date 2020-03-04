import os
import shutil

class tmpFileTracker:
    ''' Track tmp files created during test and remove those files'''
    
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def pre_walk(self):
        self.pre_list = []
        for root, dirs, files in os.walk(self.root_dir):
            for name in dirs:
                self.pre_list.append(os.path.join(root, name))
        return self

    def post_walk(self):    
        self.post_list = []
        for root, dirs, files in os.walk(self.root_dir):
            for name in dirs: 
                self.post_list.append(os.path.join(root, name))
        return self

    def rm_tracked_files(self):
        diff = list(set(self.post_list).difference(set(self.pre_list)))
        for item in diff:
            path = os.path.join(self.root_dir, item)
            shutil.rmtree(path)
        return self
