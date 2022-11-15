from __future__ import annotations

from PIL import Image

img_1 = Image.open("depthTreeTuning_R2.png")
img_2 = Image.open("depthTreeTuning_RMSE.png")


img_01_size = img_1.size
img_02_size = img_2.size


new_image = Image.new(
    "RGB",
    (2 * img_01_size[0], 1 * img_01_size[1]),
    (250, 250, 250),
)

new_image.paste(img_1, (0, 0))
new_image.paste(img_2, (img_01_size[0], 0))

new_image.save("merged-results2x1.png", "PNG")
new_image.show()
