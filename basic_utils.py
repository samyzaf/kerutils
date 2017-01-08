import sys
import os
import types
import re
import fnmatch
import inspect
import time
import datetime

def read_file(file):
    f=open(file,"r")
    text=f.read()
    f.close()
    return text

def write_file(file, data):
    f=open(file,"w")
    f.write(data)
    f.close()
    return file

def append_file(file, *data_args):
    f=open(file,"a+")
    for data in data_args:
        f.write(data)
    f.close()
    return file

def realpath(path):
    return os.path.realpath(os.path.expanduser(path))

def strmatch(str, pattern):
    return fnmatch.fnmatch(str,pattern)

def source(file):
    execfile(file)

def list_loaded_modules(pattern="*"):
    print "Modules path: " + str(sys.path)
    print ""
    print "Loaded Modules: "
    for pkg,detail in sys.modules.items():
        if fnmatch.fnmatch(pkg, pattern):
            print pkg, " -> ", detail

def list_functions(module):
    l = []
    for key,value in module.__dict__.items():
        if type(value) is types.BuiltinFunctionType:
            l.append(value)
    return l

def functions_info(module):
    "list all methods in module"
    for k,v in module.__dict__.items():
        if type(v) == types.BuiltinFunctionType or \
        type(v) == types.BuiltinMethodType or \
        type(v) == types.FunctionType or \
        type(v) == types.MethodType:
            print '%-20s: %r' % (k,type(v))

def code_info(obj):
    try:
        source = inspect.getsource(obj)
    except:
        source = "NONE"

    try:
        file = inspect.getabsfile(obj)
    except:
        try:
            file = inspect.getfile(obj)
        except:
            file = "NA"

    try:
        argspec = inspect.getargspec(obj)
    except:
        argspec = "NA"

    try:
        doc = inspect.getdoc(obj)
    except:
        doc = "NA"

    try:
        comments = inspect.getcomments(obj)
    except:
        comments = "NA"

    try:
        module = inspect.getmodule(obj)
    except:
        module = "NA"

    res  = "NAME:     " + obj.__name__      + "\n"
    res += "ARGSPEC:  " + str(argspec)      + "\n"
    res += "DOC:      " + str(doc)          + "\n"
    res += "FILE:     " + str(file)         + "\n"
    res += "MODULE:   " + module.__name__   + "\n"
    res += "COMMENTS: " + str(comments)     + "\n"
    res += "SOURCE:   " + "\n" + str(source)   + "\n"

    print res

def convert_timestr_seconds(timestr):
    arr = timestr.split()
    month = arr[1]
    day = int(arr[2])
    hour, minutes, secs = map(int, arr[3].split(':'))
    year = int(arr[4])
    date = '%d %s %d %d:%d:%d' % (day, month, year, hour, minutes, secs)
    date = datetime.datetime.strptime(date, '%d %b %Y %H:%M:%S')
    #diff = time.time() - time.mktime(date.timetuple())
    seconds = time.mktime(date.timetuple())
    return seconds

def current_time(fmt='%Y-%m-%d %H:%M:%S'):
    t = datetime.datetime.strftime(datetime.datetime.now(), fmt)
    return t

def memory_usage(pid=0):
    import psutil
    if pid == 0:
        pid = os.getpid()
    p = psutil.Process(pid)
    m = p.memory_info()
    vms = "%.2fM" % (m.vms / (1024.0**2))
    rss = "%.2fM" % (m.rss / (1024.0**2))
    return vms, rss

def convert_bytes(n):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i+1)*10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.2f%s' % (value, s)
    return "%sB" % n

def split_file_path(path):
    List = []
    while os.path.basename(path):
        List.append( os.path.basename(path) )
        path = os.path.dirname(path)
    List.reverse()
    return List

def pause(msg="Hit any key to continue ..."):
    reply=raw_input(msg)
    return reply

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        state = ["%s = %r" % (attribute, value) for (attribute,value) in self.__dict__.items()]
        return '\n'.join(state)
# that's it!  Now, you can create a Bunch
# whenever you want to group a few variables:
#
#    point = Bunch(datum=y, squared=y*y, coord=x)
#
# and of course you can read/write the named
# attributes you just created, add others, del
# some of them, etc, etc:
#    if point.squared > threshold:
#        point.isok = 1



