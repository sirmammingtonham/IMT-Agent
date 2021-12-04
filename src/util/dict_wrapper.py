import logging
from collections import defaultdict

class DictLogger(defaultdict):
    def __init__(self, *args, **kwargs):
        super(DictLogger, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getitem__(self, item):
        return super().__getitem__(item)

    def __setitem__(self, item, value):
        if super().__contains__(item):
            logging.debug('BRUH')
        return super().__setitem__(item, value)
