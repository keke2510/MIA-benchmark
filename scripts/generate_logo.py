from PIL import Image, ImageDraw
import numpy as np
import math

S = 1024
CX = CY = S / 2

# 1) 蓝->青 对角线渐变
c1 = np.array([37, 99, 235, 255], dtype=np.float64)   # #2563EB
c2 = np.array([6, 182, 212, 255], dtype=np.float64)   # #06B6D4
x, y = np.meshgrid(np.arange(S), np.arange(S))
t = (x + y) / (2.0 * (S - 1)); t = t[..., None]
grad = (c1 * (1 - t) + c2 * t).astype(np.uint8)
base = Image.fromarray(grad, 'RGBA')

# 2) 径向高光
dist = np.sqrt((x - S*0.25)**2 + (y - S*0.25)**2)
h = np.clip(1 - dist / S, 0, 1) ** 2 * 0.35
hl = np.zeros((S, S, 4), dtype=np.uint8)
hl[..., 0] = hl[..., 1] = hl[..., 2] = 255
hl[..., 3] = (h * 255).astype(np.uint8)
base = Image.alpha_composite(base, Image.fromarray(hl, 'RGBA'))

# 3) 圆角方形 + 描边
mask = Image.new('L', (S, S), 0)
md = ImageDraw.Draw(mask)
R = 220
md.rounded_rectangle([40, 40, S-40, S-40], radius=R, fill=255)
result = Image.composite(base, Image.new('RGBA', (S, S), (0,0,0,0)), mask)
draw = ImageDraw.Draw(result)
draw.rounded_rectangle([40, 40, S-40, S-40], radius=R, outline=(255,255,255,70), width=6)

# 4) 同心圆环
for rr, alpha in [(380, 26), (320, 18)]:
    draw.ellipse([CX-rr, CY-rr, CX+rr, CY+rr], outline=(255,255,255,alpha), width=3)

# 5) 五维雷达
R5 = 210
angles = [90 + 72*i for i in range(5)]
pts = [(CX + R5*math.cos(math.radians(a)), CY - R5*math.sin(math.radians(a))) for a in angles]
for p in pts:
    draw.line([(CX, CY), p], fill=(255,255,255,110), width=6)
draw.polygon(pts, fill=(255,255,255,28))
draw.line(pts + [pts[0]], fill=(255,255,255,210), width=8, joint='curve')

# 6) 5 个实心彩色顶点（黄 绿 粉 紫 橙 + 白环）
dot_colors = [(250, 204, 21), (74, 222, 128), (244, 114, 182), (192, 132, 252), (251, 146, 60)]
for p, dc in zip(pts, dot_colors):
    draw.ellipse([p[0]-30, p[1]-30, p[0]+30, p[1]+30], fill=dc, outline=(255,255,255,255), width=7)

# 7) 中心小五边形（白色实心）
r_in = 58
pts_in = [(CX + r_in*math.cos(math.radians(a+36)), CY - r_in*math.sin(math.radians(a+36))) for a in angles]
draw.polygon(pts_in, fill=(255,255,255,240))

# 8) 小星点点缀
def sparkle(px, py, r, alpha):
    draw.line([(px-r, py), (px+r, py)], fill=(255,255,255,alpha), width=4)
    draw.line([(px, py-r), (px, py+r)], fill=(255,255,255,alpha), width=4)
sparkle(180, 230, 26, 150); sparkle(830, 320, 18, 120)
sparkle(210, 790, 20, 130); sparkle(800, 760, 24, 140)

result.save('assets/logo.png')
print('saved')
