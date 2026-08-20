from PIL import Image, ImageDraw, ImageFont
import numpy as np

W, H = 2200, 700
C1 = (11, 18, 33)
C2 = (30, 58, 138)
CYAN = (34, 211, 238)

c1 = np.array(list(C1)+[255], dtype=np.float64)
c2 = np.array(list(C2)+[255], dtype=np.float64)
x, y = np.meshgrid(np.arange(W), np.arange(H))
t = (x/W + y/H) / 2.0
t = t[..., None]
grad = (c1*(1-t) + c2*t).astype(np.uint8)
banner = Image.fromarray(grad, 'RGBA')

glow = np.zeros((H, W, 4), dtype=np.uint8)
d = np.sqrt((x - W*0.85)**2 + (y - H*0.85)**2)
g = np.clip(1 - d/(W*0.7), 0, 1)**2
glow[..., 0] = CYAN[0]; glow[..., 1] = CYAN[1]; glow[..., 2] = CYAN[2]
glow[..., 3] = (g * 90).astype(np.uint8)
glow_img = Image.fromarray(glow, 'RGBA')
banner.paste(glow_img, (0, 0), glow_img)

draw = ImageDraw.Draw(banner)

# logo 区域（网格线避开这里）
IX0, IY0, IX1, IY1 = 100, 125, 550, 575

# 网格竖线：避开 logo 区域
for i in range(1, 12):
    xx = i * 200
    if IX0 <= xx <= IX1:
        draw.line([(xx, 0), (xx, IY0)], fill=(255,255,255,10), width=1)
        draw.line([(xx, IY1), (xx, H)], fill=(255,255,255,10), width=1)
    else:
        draw.line([(xx, 0), (xx, H)], fill=(255,255,255,10), width=1)

# 网格横线：避开 logo 区域
for j in range(1, 4):
    yy = j * 200
    if IY0 <= yy <= IY1:
        draw.line([(0, yy), (IX0, yy)], fill=(255,255,255,10), width=1)
        draw.line([(IX1, yy), (W, yy)], fill=(255,255,255,10), width=1)
    else:
        draw.line([(0, yy), (W, yy)], fill=(255,255,255,10), width=1)

draw.rectangle([0, 0, W, 6], fill=CYAN+(255,))

def sparkle(px, py, r, alpha):
    draw.line([(px-r, py), (px+r, py)], fill=(255,255,255,alpha), width=4)
    draw.line([(px, py-r), (px, py+r)], fill=(255,255,255,alpha), width=4)
sparkle(70, 90, 20, 100); sparkle(W-90, 80, 16, 90)
sparkle(120, H-100, 18, 90); sparkle(W-200, H-80, 14, 80)

icon = Image.open('assets/logo.png').convert('RGBA').resize((450, 450), Image.LANCZOS)
banner.paste(icon, (100, 125), icon)

title_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 180)
sub_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 46)
draw.text((660, 180), "MIA-Bench", font=title_font, fill=(255,255,255,255))
tb = draw.textbbox((660, 180), "MIA-Bench", font=title_font)
draw.rectangle([tb[0], tb[3]+10, tb[2], tb[3]+18], fill=CYAN+(255,))
draw.text((660, tb[3]+40), "Benchmarking Membership Inference Attacks", font=sub_font, fill=(226,232,240,255))
draw.text((660, tb[3]+100), "on Machine Unlearning", font=sub_font, fill=(226,232,240,255))

banner.save('assets/banner.png')
print('saved')
