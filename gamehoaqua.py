import pygame
import sys
import random
import time
from pygame.locals import *

# Khởi tạo game
WINDOWWIDTH = 400
WINDOWHEIGHT = 500
pygame.init()
w = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))

# Màu sắc và phông chữ
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Tải ảnh nền và quả
BG = pygame.image.load('bg2.jpg')
BG = pygame.transform.scale(BG, (WINDOWWIDTH, WINDOWHEIGHT))

# Kích thước của các quả
fruit_size = (40, 50)

# Tải ảnh quả và xóa phông trắng
tao = pygame.image.load('tao.jpg').convert()
tao = pygame.transform.scale(tao, fruit_size)
tao.set_colorkey((255, 255, 255))

cam = pygame.image.load('cam.jpg').convert()
cam = pygame.transform.scale(cam, fruit_size)
cam.set_colorkey((255, 255, 255))

xoai = pygame.image.load('xoai.jpg').convert()
xoai = pygame.transform.scale(xoai, fruit_size)
xoai.set_colorkey((255, 255, 255))

# Tải ảnh quả bom và xóa phông trắng, đồng thời chỉnh kích thước
bom = pygame.image.load('bom.png').convert()
bom = pygame.transform.scale(bom, fruit_size)
bom.set_colorkey((255, 255, 255))

# Biến game
FPS = 60
fpsClock = pygame.time.Clock()
diem = 0
mang_song = 100  # Thêm mạng sống
toc_do = 2  # Tốc độ chung của các quả
toc_do_bom = 2  # Tốc độ riêng cho quả bom
bom_tan_suat = 1  # Khoảng thời gian trước khi quả bom rơi

ytao, ycam, yxoai, ybom = 0, 0, 0, 0
x_tao, x_cam, x_xoai, x_bom = random.randint(100, WINDOWWIDTH - fruit_size[0]), random.randint(100, WINDOWWIDTH - fruit_size[0]), random.randint(100, WINDOWWIDTH - fruit_size[0]), random.randint(100, WINDOWWIDTH - fruit_size[0])

game_over = False


# Hàm khởi động lại vị trí quả
def reset_fruit_positions():
    global ytao, ycam, yxoai, ybom, x_tao, x_cam, x_xoai, x_bom
    ytao = 0
    ycam = random.randint(-500, -100)
    yxoai = random.randint(-1000, -500)
    ybom = random.randint(-2000, -1500)
    # Cập nhật vị trí rơi ngẫu nhiên cho các quả
    x_tao = random.randint(100, WINDOWWIDTH - fruit_size[0])
    x_cam = random.randint(100, WINDOWWIDTH - fruit_size[0])
    x_xoai = random.randint(100, WINDOWWIDTH - fruit_size[0])
    x_bom = random.randint(100, WINDOWWIDTH - fruit_size[0])

reset_fruit_positions()


# Hàm vẽ màn hình Start
def show_start_screen():
    w.fill(WHITE)
    font = pygame.font.SysFont('Arial', 50)
    text = font.render('Press START to Play', True, RED)
    w.blit(text, (WINDOWWIDTH // 2 - text.get_width() // 2, WINDOWHEIGHT // 2 - text.get_height() // 2))
    pygame.display.update()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Nhấn Enter để bắt đầu
                    waiting = False


# Hàm chơi game
def game():
    global diem, mang_song, ytao, ycam, yxoai, ybom, game_over, x_tao, x_cam, x_xoai, x_bom, bom_tan_suat,toc_do
    mang_song = 100
    diem = 0
    reset_fruit_positions()
    game_over = False
    bom_counter = 0  # Biến đếm để quản lý tần suất rơi bom

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Kiểm tra xem người chơi có bấm vào quả táo, cam, xoài, hay bom không
                if x_tao < x < x_tao + fruit_size[0] and ytao < y < ytao + fruit_size[1]:
                    diem += 5
                    ytao = 0  # Reset vị trí táo
                    x_tao = random.randint(150, WINDOWWIDTH - fruit_size[0])  # Vị trí rơi mới
                if x_cam < x < x_cam + fruit_size[0] and ycam < y < ycam + fruit_size[1]:
                    diem += 10
                    ycam = 0  # Reset vị trí cam
                    x_cam = random.randint(150, WINDOWWIDTH - fruit_size[0])  # Vị trí rơi mới
                if x_xoai < x < x_xoai + fruit_size[0] and yxoai < y < yxoai + fruit_size[1]:
                    diem += 15
                    yxoai = 0  # Reset vị trí xoài
                    x_xoai = random.randint(150, WINDOWWIDTH - fruit_size[0])  # Vị trí rơi mới
                if x_bom < x < x_bom + fruit_size[0] and ybom < y < ybom + fruit_size[1]:
                    mang_song -= 1  # Trừ mạng sống nếu ăn phải bom
                    ybom = 0
                    x_bom = random.randint(150, WINDOWWIDTH - fruit_size[0])  # Vị trí rơi mới

        # Cập nhật vị trí quả
        ytao += toc_do
        ycam += toc_do + 1
        yxoai += toc_do + 2
        toc_do += 0.0005
        # Quản lý tần suất rơi của quả bom
        bom_counter += 1
        if bom_counter >= bom_tan_suat:
            ybom += toc_do_bom
            if ybom > WINDOWHEIGHT:
                ybom = 0
                x_bom = random.randint(150, WINDOWWIDTH - fruit_size[0])  # Đổi vị trí rơi của quả bom
            bom_counter = 0  # Reset bộ đếm

        # Reset vị trí khi quả ra khỏi màn hình
        if ytao > WINDOWHEIGHT:
            ytao = 0
            x_tao = random.randint(100, WINDOWWIDTH - fruit_size[0])
            mang_song -= 1
        if ycam > WINDOWHEIGHT:
            ycam = 0
            x_cam = random.randint(100, WINDOWWIDTH - fruit_size[0])
            mang_song -= 1
        if yxoai > WINDOWHEIGHT:
            yxoai = 0
            x_xoai = random.randint(100, WINDOWWIDTH - fruit_size[0])
            mang_song -= 1

        # Kết thúc trò chơi khi hết mạng
        if mang_song <= 0:
            game_over = True
            break

        # Vẽ nền và quả
        w.blit(BG, (0, 0))
        w.blit(tao, (x_tao, ytao))
        w.blit(cam, (x_cam, ycam))
        w.blit(xoai, (x_xoai, yxoai))
        w.blit(bom, (x_bom, ybom))

        # In điểm và mạng sống
        font = pygame.font.SysFont('Arial', 30)
        text = font.render(f'Điểm: {diem}', True, (255, 0, 0))
        w.blit(text, (50, 50))

        text_mang = font.render(f'Mạng sống: {mang_song}', True, (255, 0, 0))
        w.blit(text_mang, (50, 80))

        pygame.display.update()
        fpsClock.tick(FPS)


# Vòng lặp chính của game
while True:
    if game_over:
        show_start_screen()
    show_start_screen()
    game()
