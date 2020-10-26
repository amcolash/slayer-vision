# import the opencv library
import cv2
from mss import mss
import numpy as np
import subprocess
import time

def resize(path):
  img = cv2.imread(path, 0)
  return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

def find(img_rgb, img_gray, template, threshold = 0.75):
  found = []
  w, h = template.shape[::-1]
  res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
  loc = np.where( res >= threshold)
  for pt in zip(*loc[::-1]):
    point = [pt, (pt[0] + w, pt[1] + h)]
    cv2.rectangle(img_rgb, point[0], point[1], (0,0,255), 2)
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

coin = resize('images/coin_1.png')
bee = resize('images/bee_1.png')
player = resize('images/player_1.png')
boost = resize('images/boost.png')
block = resize('images/orange_block.png')

with mss() as sct:
  while(True):
    img_rgb = np.array(sct.grab(monitor))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    coins = find(img_rgb, img_gray, coin, 0.5)

    should_jump = False

    for c in coins:
      if c[0][0] < 150:
        should_jump = True

    if should_jump:
      subprocess.check_output(["xdotool", "mousemove", str(int(x + w / 2)), str(int(y + h / 2)), "click", "1"])

    cv2.imshow('frame', img_rgb)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
