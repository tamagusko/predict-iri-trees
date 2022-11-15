from __future__ import annotations

from PIL import Image

img_1 = Image.open("DT.png")
img_2 = Image.open("RF.png")
img_3 = Image.open("XGBoost.png")
img_4 = Image.open("treesComparison.png")

img_01_size = img_1.size
img_02_size = img_2.size
img_03_size = img_3.size
img_02_size = img_4.size

new_image = Image.new(
    "RGB",
    (2 * img_01_size[0], 2 * img_01_size[1]),
    (250, 250, 250),
)

new_image.paste(img_1, (0, 0))
new_image.paste(img_2, (img_01_size[0], 0))
new_image.paste(img_3, (0, img_01_size[1]))
new_image.paste(img_4, (img_01_size[0], img_01_size[1]))

new_image.save("merged-results.png", "PNG")
new_image.show()
