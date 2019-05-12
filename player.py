from collections import deque, namedtuple

Frame = namedtuple('Frame', ['index', 'cells', 'image'])


class Player():
    def __init__(self):
        self.count = 0
        self.last_frames = deque(maxlen=30)

    def play(self, cells):
        image = get_artificial_image(cells)
        frame = Frame(self.count, cells, image)

        self.last_frames.append(frame)
        self.count += 1
