from Queue import Queue, Empty
from threading import Thread, Event


class Signal(object):
    def __init__(self, name=None):
        self.name = name
        self.data = {}


class EventEngine(object):
    def __init__(self):
        self.__queue = Queue()
        self.__thread = Thread(target=self.__run)
        self.__active = Event()
        self.__handlers = {}

    def __run(self):
        while self.__active.is_set():
            try:
                event = self.__queue.get(timeout=1.0)
                self.__process(event)
            except Empty:
                pass

    def __process(self, event):
        if event.name in self.__handlers:
            [handler(event) for handler in self.__handlers[event.name]]

    def start(self):
        self.__active.set()
        self.__thread.start()

    def stop(self):
        self.__active.clear()
        self.__thread.join()

    def register(self, name, handler):
        if name not in self.__handlers.keys():
            self.__handlers[name] = [handler]
        else:
            self.__handlers[name].append(handler)

    def unregister(self, name, handler):
        if name in self.__handlers.keys() and handler in self.__handlers[name]:
            self.__handlers[name].remove(handler)

    def put(self, event):
        self.__queue.put(event)
