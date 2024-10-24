import pygame
import random
import sys

# Khởi tạo game
pygame.init()
WINDOWWIDTH = 1000
WINDOWHEIGHT = 600
w = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
pygame.display.set_caption('Flappy Bird')

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Tải ảnh
BG = pygame.image.load('background.jpg')
BG = pygame.transform.scale(BG, (WINDOWWIDTH, WINDOWHEIGHT))

BIRD_IMG = pygame.image.load('bird_right.jpg').convert()
BIRD_IMG = pygame.transform.scale(BIRD_IMG, (30, 40))
BIRD_IMG.set_colorkey((255, 255, 255))

PIPE_IMG = pygame.image.load('cột.png').convert()
PIPE_IMG = pygame.transform.scale(PIPE_IMG, (150, 500))
PIPE_IMG.set_colorkey((255, 255, 255))

# Thông số trò chơi
FPS = 60
GRAVITY = 0.5
BIRD_JUMP = -7
PIPE_SPEED = 4
PIPE_GAP = 150

# Biến
bird_y = WINDOWHEIGHT // 2
bird_x = 100
bird_velocity = 0
pipes = []
pipe_frequency = 1500  # 1.5 giây xuất hiện 1 cột
score = 0
game_over = False

# Đồng hồ thời gian
clock = pygame.time.Clock()
last_pipe_time = pygame.time.get_ticks()

# Hàm tạo cột
def create_pipe():
    height = random.randint(150, 400)
    bottom_pipe = PIPE_IMG.get_rect(midtop=(WINDOWWIDTH + 100, height))
    top_pipe = PIPE_IMG.get_rect(midbottom=(WINDOWWIDTH + 100, height - PIPE_GAP))
    return bottom_pipe, top_pipe

# Hàm di chuyển cột
def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= PIPE_SPEED
    return [pipe for pipe in pipes if pipe.right > 0]

# Hàm vẽ cột
def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= WINDOWHEIGHT:
            w.blit(PIPE_IMG, pipe)
        else:  # Ống trên cần phải xoay ngược lại
            flip_pipe = pygame.transform.flip(PIPE_IMG, False, True)
            w.blit(flip_pipe, pipe)

# Hàm kiểm tra va chạm
def check_collision(pipes):
    global game_over
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            game_over = True
    if bird_rect.top <= -100 or bird_rect.bottom >= WINDOWHEIGHT:
        game_over = True

# Hàm chính của game
# Hàm chính của game
def game():
    global bird_y, bird_velocity, pipes, score, game_over

    bird_y = WINDOWHEIGHT // 2
    bird_velocity = 0
    pipes = []
    score = 0
    game_over = False
    last_pipe_time = pygame.time.get_ticks()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird_velocity = BIRD_JUMP

        # Trọng lực
        bird_velocity += GRAVITY
        bird_y += bird_velocity

        # Cập nhật vị trí và hình chữ nhật của chim
        bird_rect = BIRD_IMG.get_rect(center=(bird_x, bird_y))

        # Tạo ống mới
        current_time = pygame.time.get_ticks()
        if current_time - last_pipe_time > pipe_frequency:
            pipes.extend(create_pipe())
            last_pipe_time = current_time

        # Di chuyển và vẽ ống
        pipes = move_pipes(pipes)

        # Kiểm tra va chạm
        for pipe in pipes:
            if bird_rect.colliderect(pipe):  # Kiểm tra va chạm giữa chim và cột
                game_over = True
        if bird_rect.top <= -100 or bird_rect.bottom >= WINDOWHEIGHT:
            game_over = True

        # Tăng điểm nếu chim vượt qua ống
        for pipe in pipes:
            if pipe.centerx == bird_x:
                score += 1

        # Vẽ nền, chim và ống
        w.blit(BG, (0, 0))
        w.blit(BIRD_IMG, bird_rect)
        draw_pipes(pipes)

        # Hiển thị điểm
        font = pygame.font.SysFont('Arial', 30)
        text = font.render(f'Score: {score}', True, BLACK)
        w.blit(text, (10, 10))

        pygame.display.update()
        clock.tick(FPS)

    return score

# Vòng lặp chính
while True:
    w.fill(WHITE)
    font = pygame.font.SysFont('Arial', 50)
    text = font.render('Press SPACE to Play', True, RED)
    w.blit(text, (WINDOWWIDTH // 2 - text.get_width() // 2, WINDOWHEIGHT // 2 - text.get_height() // 2))
    pygame.display.update()

    # Đợi người chơi nhấn phím SPACE để bắt đầu
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    # Bắt đầu trò chơi
    final_score = game()

    # Sau khi game over
    w.fill(WHITE)
    text = font.render(f'Game Over! Final Score: {final_score}', True, RED)
    w.blit(text, (WINDOWWIDTH // 2 - text.get_width() // 2, WINDOWHEIGHT // 2 - text.get_height() // 2))
    pygame.display.update()
    pygame.time.wait(2000)
