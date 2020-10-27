# import the opencv library
import cv2
from mss import mss
import numpy as np
from os import path
import subprocess
import threading
import time

def resize(f):
  if not path.exists(f):
    print("Could not find the file " + f)
    raise IOError
  img = cv2.imread(f, 0)
  return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

def find(img_gray, draw_img, templates, threshold = 0.75):
  found = []

  for template in templates:
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
      point = [pt, (pt[0] + w, pt[1] + h)]
      cv2.rectangle(draw_img, point[0], point[1], (0,0,255), 2)
      found.append(point)

  return found

fullW = 2280
fullH = 1080

window = subprocess.check_output(["xdotool", "search", "--name", "Pixel"], universal_newlines=True)

geom = subprocess.check_output(["xdotool", "getwindowgeometry", "--shell", window], universal_newlines=True)
x = y = w = h = 0
for l in geom.splitlines():
  if l.startswith("WIDTH"):
    w = int(l.replace("WIDTH=", ""))
  if l.startswith("HEIGHT"):
    h = int(l.replace("HEIGHT=", ""))
  if l.startswith("X"):
    x = int(l.replace("X=", ""))
  if l.startswith("Y"):
    y = int(l.replace("Y=", ""))

subprocess.check_output(["xdotool", "windowactivate", window])

monitor = {"top": y, "left": x, "width": w, "height": h}

# scale and load test images
scale = ((w / fullW) + (h / fullH)) / 2

coin1 = resize('images/coin_1.png')
coin2 = resize('images/coin_2.png')
ruby = resize('images/ruby_1.png')
orange_block = resize('images/orange_block.png')
bee = resize('images/bee_1.png')
boost_img = resize('images/boost.png')
bonus_img = resize('images/bonus.png')

boost_timeout = time.time() - 15
bonus_count = 0

title = "frame"
cv2.namedWindow(title)
cv2.moveWindow(title, x - 10, 70)

with mss() as sct:
  while(True):
    frame = time.time()
    screenshot = time.time()

    img_rgb = np.array(sct.grab(monitor))
    img_gray = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_BGR2GRAY)
    draw_img = img_rgb.copy()

    screenshot = 'screen ' + str(round((time.time() - screenshot) * 1000, 2))
    image = cv2.putText(draw_img, screenshot, (w - 140, h - 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # TODO: make thresholds relative to scale
    coin_thresh_x_left = 60
    coin_thresh_x_right = 180
    coin_thresh_x_boost = int(coin_thresh_x_right + 250 * (1 - (time.time() - boost_timeout) / 10))
    coin_thresh_y = 175
    cv2.line(draw_img, (coin_thresh_x_left, 0), (coin_thresh_x_left, h), (0,0,255), 2)
    cv2.line(draw_img, (coin_thresh_x_right, 0), (coin_thresh_x_right, h), (0,0,255), 2)
    cv2.line(draw_img, (0, coin_thresh_y), (w, coin_thresh_y), (0,0,255), 2)

    search_w = max(350, coin_thresh_x_boost)
    search_h = 250
    search_region = img_rgb[0:search_h, 0:search_w]
    img_gray_region = cv2.cvtColor(search_region.copy(), cv2.COLOR_BGR2GRAY)
    cv2.rectangle(draw_img, (0, 0), (search_w, search_h), (255,0,0), 2)

    find_coins = time.time()
    coins = find(img_gray_region, draw_img, [coin1, coin2, ruby], 0.5)
    find_coins = 'coins ' + str(round((time.time() - find_coins) * 1000, 2))
    image = cv2.putText(draw_img, find_coins, (w - 140, h - 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    find_blocks = time.time()
    block = find(img_gray_region, draw_img, [orange_block])
    find_blocks = 'blocks ' + str(round((time.time() - find_blocks) * 1000, 2))
    image = cv2.putText(draw_img, find_blocks, (w - 140, h - 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    find_enemies = time.time()
    enemies = find(img_gray_region, draw_img, [bee], 0.5)
    find_enemies = 'enemies ' + str(round((time.time() - find_enemies) * 1000, 2))
    image = cv2.putText(draw_img, find_enemies, (w - 140, h - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    logic = time.time()

    should_jump = False
    is_boost = time.time() - boost_timeout < 10
    boost_enabled = time.time() - boost_timeout >= 14

    for c in coins:
      if c[0][0] > coin_thresh_x_left and c[0][1] < coin_thresh_y:
        if (is_boost and c[0][0] < coin_thresh_x_boost) or c[0][0] < coin_thresh_x_right:
          cv2.rectangle(draw_img, c[0], c[1], (0,255,0), 2)
          should_jump = True

    for e in enemies:
      if e[0][0] > coin_thresh_x_left:
        if (is_boost and e[0][0] < coin_thresh_x_boost) or e[0][0] < coin_thresh_x_right:
          cv2.rectangle(draw_img, e[0], e[1], (0,255,0), 2)
          should_jump = True

    block_thresh = 100
    cv2.line(draw_img, (block_thresh, 0), (block_thresh, h), (0,0,255), 2)

    for b in block:
      if b[0][0] < block_thresh:
        cv2.rectangle(draw_img, b[0], b[1], (255,0,0), 2)
        should_jump = True

    if should_jump:
      subprocess.Popen(f"xdotool mousemove {str(int(x + w / 2))} {str(int(y + h / 2))} mousedown 1 sleep 0.15 mouseup 1", shell=True)

    if is_boost:
      cv2.line(draw_img, (coin_thresh_x_boost, 0), (coin_thresh_x_boost, h), (0,0,255), 2)

    # TODO: make this work better since coins reduce cooldown
    # only boost when there is not a frenzy
    if len(coins) < 10 and boost_enabled:
      boost = find(img_gray, draw_img, [boost_img], 0.65)

      if len(boost) > 0:
        b = boost[0]
        subprocess.Popen(f"xdotool mousemove {str(int(x + (b[0][0] + b[1][0]) / 2))} {str(int(y + (b[0][1] + b[1][1]) / 2))} click 1", shell=True)
        boost_timeout = time.time()

    # For now, just run the bonus and fail
    bonus_count = bonus_count + 1
    if bonus_count > 150:
      bonus_count = 0
      bonus = find(img_gray, draw_img, [bonus_img])

      if len(bonus) > 0:
        b = bonus[0]
        subprocess.Popen(f"xdotool mousemove {str(int(x + (b[0][0] + b[1][0]) / 2))} {str(int(y + (b[0][1] + b[1][1]) / 2))} click 1", shell=True)

    logic = 'logic ' + str(round((time.time() - logic) * 1000, 2))
    image = cv2.putText(draw_img, logic, (w - 140, h - 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    fps = 'fps ' + str(round(1 / (time.time() - frame), 2))
    image = cv2.putText(draw_img, fps, (w - 140, h - 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow(title, draw_img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Destroy all the windows
cv2.destroyAllWindows()
