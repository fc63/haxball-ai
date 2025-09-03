import sys, pygame
import time
from typing import List, Union, Optional
from simulator import create_start_conditions, Vector, GamePlay


size = width, height = 900, 520
center = (width // 2, height // 2 + 30)
black = 105, 150, 90


def format_time(seconds: Union[int, float]) -> str:
    mins = int(seconds // 60) % 60
    hours = int(seconds // 3600)
    secs = int(seconds % 60)
    return "{:0>2d}:{:0>2d}:{:0>2d}".format(hours, mins, secs)


def draw_frame(screen, gameplay: GamePlay, reward: Optional[float] = None, ret: Optional[float] = None):
    font = pygame.font.SysFont("monospace", 44)
    font_small = pygame.font.SysFont("monospace", 18)
    screen.fill(black)

    # Linee
    pygame.draw.rect(screen, (220, 220, 220), (center[0] - 370, center[1] - 170, 370 * 2, 170 * 2), 2)
    pygame.draw.circle(screen, (220, 220, 220), center, 75, 2)
    pygame.draw.line(screen, (220, 220, 220), (center[0], center[1] - 170), (center[0], center[1] + 170), 2)

    # Rosso
    pygame.draw.circle(screen, (225, 100, 90),
                       (center[0] + int(round(gameplay.wa.K[5].a.x)), center[1] + int(round(gameplay.wa.K[5].a.y))),
                       gameplay.wa.K[5].la, 0)
    if gameplay.Pa.D[1].bc:
        pygame.draw.circle(screen, (255, 255, 255),
                           (center[0] + int(round(gameplay.wa.K[5].a.x)), center[1] + int(round(gameplay.wa.K[5].a.y))),
                           gameplay.wa.K[5].la, 1)
    else:
        pygame.draw.circle(screen, (0, 0, 0),
                           (center[0] + int(round(gameplay.wa.K[5].a.x)), center[1] + int(round(gameplay.wa.K[5].a.y))),
                           gameplay.wa.K[5].la, 1)
    # White indicator near red when holding kick
    if getattr(gameplay.Pa.D[1], 'kick_indicator', None) is not None:
        ki = gameplay.Pa.D[1].kick_indicator
        pygame.draw.circle(screen, (255, 255, 255),
                           (center[0] + int(round(ki.x)), center[1] + int(round(ki.y))), 3, 0)

    # Blu
    pygame.draw.circle(screen, (100, 90, 245),
                       (center[0] + int(round(gameplay.wa.K[6].a.x)), center[1] + int(round(gameplay.wa.K[6].a.y))),
                       gameplay.wa.K[6].la, 0)
    if gameplay.Pa.D[2].bc:
        pygame.draw.circle(screen, (255, 255, 255),
                           (center[0] + int(round(gameplay.wa.K[6].a.x)), center[1] + int(round(gameplay.wa.K[6].a.y))),
                           gameplay.wa.K[6].la, 1)
    else:
        pygame.draw.circle(screen, (0, 0, 0),
                           (center[0] + int(round(gameplay.wa.K[6].a.x)), center[1] + int(round(gameplay.wa.K[6].a.y))),
                           gameplay.wa.K[6].la, 1)
    # White indicator near blue when holding kick
    if getattr(gameplay.Pa.D[2], 'kick_indicator', None) is not None:
        ki = gameplay.Pa.D[2].kick_indicator
        pygame.draw.circle(screen, (255, 255, 255),
                           (center[0] + int(round(ki.x)), center[1] + int(round(ki.y))), 3, 0)

    # Palla
    pygame.draw.circle(screen, (220, 220, 220),
                       (center[0] + int(round(gameplay.wa.K[0].a.x)), center[1] + int(round(gameplay.wa.K[0].a.y))),
                       gameplay.wa.K[0].la, 0)
    pygame.draw.circle(screen, (50, 50, 50),
                       (center[0] + int(round(gameplay.wa.K[0].a.x)), center[1] + int(round(gameplay.wa.K[0].a.y))),
                       gameplay.wa.K[0].la + 1, 1)

    # Pali
    for i in range(4):
        pygame.draw.circle(screen, (100, 100, 100),
                           (center[0] + int(gameplay.wa.K[1 + i].a.x), center[1] + int(gameplay.wa.K[1 + i].a.y)),
                           gameplay.wa.K[1 + i].la, 0)
        pygame.draw.circle(screen, (0, 0, 0),
                           (center[0] + int(gameplay.wa.K[1 + i].a.x), center[1] + int(gameplay.wa.K[1 + i].a.y)),
                           gameplay.wa.K[1 + i].la, 1)

    # Punteggio
    text = font.render("%s - %s" % (gameplay.Kb, gameplay.Cb), True, (0, 0, 0))
    screen.blit(text, (center[0] - text.get_width() // 2, center[1] - 250))

    # Tempo
    text = font.render(format_time(gameplay.Ac), True, (0, 0, 0))
    screen.blit(text, (center[0] - 375, center[1] - 250))

    # Reward
    if reward is not None and ret is not None:
        text = font_small.render("Rew: %2.3f (Ret: %2.3f)" % (reward, ret), True, (0, 0, 0))
        screen.blit(text, (center[0] + 150, center[1] - 220))

    # Debug HUD: show player speeds and hold flags
    red_v = gameplay.wa.K[5].M
    blue_v = gameplay.wa.K[6].M
    red_speed = (red_v.x ** 2 + red_v.y ** 2) ** 0.5
    blue_speed = (blue_v.x ** 2 + blue_v.y ** 2) ** 0.5
    red_hold = getattr(gameplay.Pa.D[1], 'hold', False)
    blue_hold = getattr(gameplay.Pa.D[2], 'hold', False)
    hud1 = font_small.render(f"RED | v={red_speed:5.2f} hold={'Y' if red_hold else 'N'}", True, (0, 0, 0))
    hud2 = font_small.render(f"BLU | v={blue_speed:5.2f} hold={'Y' if blue_hold else 'N'}", True, (0, 0, 0))
    screen.blit(hud1, (10, 10))
    screen.blit(hud2, (10, 30))


if __name__ == '__main__':
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size)

    gameplay = create_start_conditions(
        posizione_palla=Vector(0, 0),
        velocita_palla=Vector(0, 0),
        posizione_blu=Vector(277.5, 0),
        velocita_blu=Vector(0, 0),
        input_blu=0,
        posizione_rosso=Vector(-277.5, 0),
        velocita_rosso=Vector(0, 0),
        input_rosso=0,
        tempo_iniziale=0,
        punteggio_rosso=0,
        punteggio_blu=0
    )

    blue_unpressed = True
    red_unpressed = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        gameplay.Pa.D[2].mb = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            gameplay.Pa.D[2].mb |= 1
        if keys[pygame.K_DOWN]:
            gameplay.Pa.D[2].mb |= 2
        if keys[pygame.K_RIGHT]:
            gameplay.Pa.D[2].mb |= 8
        if keys[pygame.K_LEFT]:
            gameplay.Pa.D[2].mb |= 4
        if keys[pygame.K_SPACE]:
            if blue_unpressed:
                gameplay.Pa.D[2].mb |= 16
                gameplay.Pa.D[2].bc = 1
            blue_unpressed = False
        else:
            gameplay.Pa.D[2].bc = 0
            blue_unpressed = True

        gameplay.Pa.D[1].mb = 0
        gameplay.Pa.D[1].bc = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            gameplay.Pa.D[1].mb |= 1
        if keys[pygame.K_s]:
            gameplay.Pa.D[1].mb |= 2
        if keys[pygame.K_d]:
            gameplay.Pa.D[1].mb |= 8
        if keys[pygame.K_a]:
            gameplay.Pa.D[1].mb |= 4
        if keys[pygame.K_LCTRL]:
            if red_unpressed:
                gameplay.Pa.D[1].mb |= 16
                gameplay.Pa.D[1].bc = 1
            red_unpressed = False
        else:
            red_unpressed = True

        draw_frame(screen, gameplay)

        # screen.blit(ball, ballrect)
        pygame.display.flip()
        clock.tick(60)
        gameplay.step(1)

        if gameplay.zb == 2:
            gameplay = create_start_conditions(
                posizione_palla=Vector(0, 0),
                velocita_palla=Vector(0, 0),
                posizione_blu=Vector(277.5, 0),
                velocita_blu=Vector(0, 0),
                input_blu=0,
                posizione_rosso=Vector(-277.5, 0),
                velocita_rosso=Vector(0, 0),
                input_rosso=0,
                tempo_iniziale=gameplay.Ac,
                punteggio_rosso=gameplay.Kb,
                punteggio_blu=gameplay.Cb,
                commincia_rosso=gameplay.Jd.sn == 't-red'
            )
