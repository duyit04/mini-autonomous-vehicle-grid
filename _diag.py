import os
import numpy as np
import matplotlib.image as mpimg
import main3d as m

app = m.CarViz3D()
app._autorot.set(False)
app.update()
app._fig.savefig("._diag.png", dpi=80)
app._on_close()

im = mpimg.imread("._diag.png")
rgb = (im[..., :3] * 255).astype(int)
print("shape", im.shape)


def near(c, tol=40):
    return int(((np.abs(rgb[..., 0] - c[0]) < tol)
                & (np.abs(rgb[..., 1] - c[1]) < tol)
                & (np.abs(rgb[..., 2] - c[2]) < tol)).sum())


print("blue car px     ~", near((59, 130, 246), 25))
print("cabin pink px   ~", near((252, 165, 165), 25))
print("obstacle gray   ~", near((71, 85, 105), 12))
print("obstacle top    ~", near((100, 116, 139), 12))
print("green goal px   ~", near((16, 185, 129), 25))
print("yellow nose px  ~", near((250, 204, 21), 25))
print("trail blue px   ~", near((56, 189, 248), 25))
print("car decoded pos ~", app.env.state_decoder(app._obs)[:2])
os.remove("._diag.png")
