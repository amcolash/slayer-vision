# import the opencv library
import cv2
from mss import mss
import numpy as np
import subprocess
import time

def resize(path):
  img = cv2.imread(path, 0)
  return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

def find(img_rgb, img_gray, draw_img, templates, threshold = 0.75):
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
coin3 = resize('images/coin_3.png')
orange_block = resize('images/orange_block.png')
boost_img = resize('images/boost.png')
bonus_img = resize('images/bonus.png')

boost_timeout = time.time() - 15
bonus_count = 0

with mss() as sct:
  while(True):
    img_rgb = np.array(sct.grab(monitor))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    draw_img = img_rgb.copy()

    coins = find(img_rgb, img_gray, draw_img, [coin1, coin2], 0.5)
    block = find(img_rgb, img_gray, draw_img, [orange_block])
    boost = find(img_rgb, img_gray, draw_img, [boost_img], 0.65)

    should_jump = False
    is_boost = time.time() - boost_timeout < 10
    boost_enabled = time.time() - boost_timeout >= 14

    # TODO: make thresholds relative to scale
    coin_thresh_x_left = 60
    coin_thresh_x_right = 180
    coin_thresh_x_boost = int(coin_thresh_x_right + 225 * (1 - (time.time() - boost_timeout) / 10))
    coin_thresh_y = 175
    cv2.line(draw_img, (coin_thresh_x_left, 0), (coin_thresh_x_left, h), (0,0,255), 2)
    cv2.line(draw_img, (coin_thresh_x_right, 0), (coin_thresh_x_right, h), (0,0,255), 2)
    cv2.line(draw_img, (0, coin_thresh_y), (w, coin_thresh_y), (0,0,255), 2)

    if is_boost:
      cv2.line(draw_img, (coin_thresh_x_boost, 0), (coin_thresh_x_boost, h), (0,0,255), 2)

    for c in coins:
      if c[0][0] > coin_thresh_x_left and c[0][1] < coin_thresh_y:
        if (is_boost and c[0][0] < coin_thresh_x_boost) or c[0][0] < coin_thresh_x_right:
          cv2.rectangle(draw_img, c[0], c[1], (0,255,0), 2)
          should_jump = True

    block_thresh = 100
    cv2.line(draw_img, (block_thresh, 0), (block_thresh, h), (0,0,255), 2)

    for b in block:
      if b[0][0] < block_thresh:
        cv2.rectangle(draw_img, b[0], b[1], (255,0,0), 2)
        should_jump = True

    # should_jump = True
    if should_jump:
      subprocess.check_output(["xdotool", "mousemove", str(int(x + w / 2)), str(int(y + h / 2)), "mousedown", "1", "sleep", "0.15", "mouseup", "1"])

    # TODO: make this work better since coins reduce cooldown
    # only boost when there is not a frenzy
    if len(coins) < 16 and len(boost) > 0 and boost_enabled:
      b = boost[0]
      subprocess.check_output(["xdotool", "mousemove", str(int(x + (b[0][0] + b[1][0]) / 2)), str(int(y + (b[0][1] + b[1][1]) / 2)), "click", "1"])
      boost_timeout = time.time()

    # For now, just run the bonus and fail
    bonus_count = bonus_count + 1
    if bonus_count > 30:
      bonus_count = 0
      bonus = find(img_rgb, img_gray, draw_img, [bonus_img])
      if len(bonus) > 0:
        b = bonus[0]
        subprocess.check_output(["xdotool", "mousemove", str(int(x + (b[0][0] + b[1][0]) / 2)), str(int(y + (b[0][1] + b[1][1]) / 2)), "click", "1"])

    cv2.imshow('frame', draw_img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
