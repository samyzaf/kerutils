import sys, time

class Progmeter:
    def __init__(self, size, **opt):
        prompt = opt.get('prompt', 'Progress: ')
        callback = opt.get('callback', None)
        self.size = size
        self.chunk = size / 100.0
        self.percent = -1
        self.index = 0
        self.done = False
        self.start = time.time()
        self.text = '\r' + prompt + '%d%%   '
        self.callback = callback

    def advance(self, n=None):
        if self.done:
            return
        if n is None:
            self.index += 1
        else:
            self.index = n
        if self.index >= self.size - 1:
            sys.stdout.write((self.text + '\n') % 100)
            sys.stdout.flush()
            self.done = True
            t = time.time() - self.start
            print "Time: %.2f seconds" % t
            return
        p = int(self.index / self.chunk)
        if p > self.percent:
            sys.stdout.write(self.text % p)
            sys.stdout.flush()
            self.percent = p
            if self.callback:
                exec(self.callback+'()')


