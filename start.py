import sys
import time

# add osmo.sessdsa to search path
sys.path.append('../osmo.sessdsa/src')

from consts import Consts
from player import Player
from world import World


if __name__ == "__main__":
    world = World()
    # For timer
    frame_delta = None
    last_tick = int(round(time.time() * 1000))

    player = Player()
    while not world.result:
        # Advance timer
        current_tick = int(round(time.time() * 1000))
        frame_delta = (current_tick - last_tick) * Consts["FPS"] / 1000
        last_tick = current_tick
        world.update(Consts["FRAME_DELTA"])
        player.play(world.cells)
        # only test 1 frame
        break
