from datetime import datetime
from typing import List

from Games.financial_model.archive.structures.Event import Event


class TimelinePointer:
    def __init__(self, value: Event or None, index: int, timeline):
        self.value = value
        self.index = index
        self.timeline = timeline

    def next(self):
        next_index = self.index + 1
        if next_index >= len(self.timeline.list):
            return TimelinePointer(None, next_index, self.timeline)

        return TimelinePointer(self.timeline.list[next_index], next_index, self.timeline)

    def previous(self):
        previous_index = self.index - 1
        if previous_index < 0:
            return TimelinePointer(None, previous_index, self.timeline)

        return TimelinePointer(self.timeline.list[previous_index], previous_index, self.timeline)


class Timeline:
    def __init__(self):
        self.list: List[Event] = []

    """
    Will return a pointer to the first list element that has a datetime >= the given timestamp.
    So if there is an event with the same timestamp it will point to that.
    If there is not, it will point to the position such an element should be placed which can be outside of the lists bound.
    """

    def current(self, timestamp: datetime) -> TimelinePointer:
        pos = len(self.list)
        for i in range(len(self.list)):
            local_timestamp = self.list[i].timestamp
            if local_timestamp >= timestamp:
                pos = i
                break
        if 0 <= pos < len(self.list):
            return TimelinePointer(self.list[pos], pos, self)
        else:
            return TimelinePointer(None, pos, self)

    """
    Returns a pointer to the last element with a timestamp < the given timestamp.
    """

    def previous(self, timestamp: datetime):
        pos = len(self.list)
        for i in range(len(self.list)):
            local_timestamp = self.list[i].timestamp
            if local_timestamp >= timestamp:
                pos = i - 1
                break
        if 0 <= pos < len(self.list):
            return TimelinePointer(self.list[pos], pos, self)
        else:
            return TimelinePointer(None, pos, self)

    """
    Returns a pointer to the first element with a timestamp > the given timestamp.
    """

    def next(self, timestamp: datetime):
        pos = len(self.list)
        for i in range(len(self.list)):
            local_timestamp = self.list[i].timestamp
            if local_timestamp > timestamp:
                pos = i
                break
        if 0 <= pos < len(self.list):
            return TimelinePointer(self.list[pos], pos, self)
        else:
            return TimelinePointer(None, pos, self)

    def add(self, event: Event):
        insert_position = self.current(event.timestamp).index

        if 0 < insert_position < len(self.list):
            if self.list[insert_position] == event.timestamp:
                self.list[insert_position] = event
                return

        self.list.insert(insert_position, event)

    def __add__(self, other):
        t = Timeline()
        t.list = self.list.copy()
        for e in other.list:
            t.add(e)
        return t
