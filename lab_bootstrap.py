# run once before you launch the test
from stb_registry import register_capture
from dp_dispatcher import register_stb           # if you route commands via DP Studio

CAP_DEV = 'video=Video (00-0 Pro Capture Quad HDMI)'   # index 1
STB_ID  = '1'

register_capture(STB_ID, CAP_DEV)
register_stb(STB_ID, capture=CAP_DEV, portview='8')    # '8' is an example port
