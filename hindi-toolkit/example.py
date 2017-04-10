from core import Core

text = "जल्‍दी से मालदार हो जाने की हवस किसे नहीं होती ?"

c = Core()
pos_tagged = c.tag_pos(text)
print(pos_tagged)