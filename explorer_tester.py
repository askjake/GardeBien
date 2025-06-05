from capture_config import dshow_name_for_port
from dp_dispatcher import register_stb
from explorer import Explorer, _ffmpeg_screenshot
import stb_registry as reg
from ui_graph import UIGraph

STB = "A"
DEV = dshow_name_for_port(128)   # video=...
PORTVIEW = "8"

register_stb(STB, capture=DEV, portview=PORTVIEW)

grab = lambda: _ffmpeg_screenshot(DEV)

with UIGraph(uri="bolt://10.74.139.250:7687",
             user="neo4j", password="mango-metal-moral-bronze-prague-8964") as g:
    Explorer(STB, grab, g).self_explore(25)
